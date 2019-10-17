from _context import former
from former import util, GTransformer

from util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import os, random, tqdm, sys, math, gzip

from radam import RAdam

# NB, the enwik8 data contains tokens from 9 to 240, but well round up to the nearest
# power of two.
NUM_TOKENS = 256
# Used for converting between nats and bits
LOG2E = math.log2(math.e)

USE_APEX = False
if USE_APEX:
    from apex import amp

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()
    
def read_dataset(path, name):
    def read_data(filename):        
        with open(os.path.join(path, filename), encoding='cp874') as file:
            X = np.frombuffer(file.read().encode('cp874'), dtype=np.uint8)
            return torch.from_numpy(X)
    
    trX = read_data('%s-train.txt' % name)
    vaX = read_data('%s-val.txt' % name)
    teX = read_data('%s-test.txt' % name)
    
    return trX, vaX, teX

def generate_sequence(arg, model, data, starter=None, gensize=1000, temp=0.5):

    with torch.no_grad():
        
        # generate some random text
        if starter is None:
            seedfr = random.randint(0, data.size(0) - arg.context)
            input = data[seedfr:seedfr + arg.context].to(torch.long)
            starter = input
        else:
            if starter.size(0) < arg.context:
                pad = torch.zeros(size=(arg.context - starter.size(0),), dtype=torch.long)
                input = torch.cat([pad, starter.to(torch.long)], dim=0)
            else:
                input = starter[starter.size(0) - arg.context:].to(torch.long)            

        if torch.cuda.is_available():
            input = input.cuda()

        input = Variable(input)

        print('[', end='', flush=True)
        for c in starter:
            print(str(bytes([c]).decode('cp874')), end='', flush=True)
        print(']', end='', flush=True)

        for _ in range(gensize):
            output = model(input[None, :])
            c = sample(output[0, -1, :], temp)
            
            try:
                print(str(bytes([c]).decode('cp874')), end='', flush=True)
            except:
                print('x', flush=True)

            input = torch.cat([input[1:], c[None]], dim=0)

        print()

def calculate_bpb(arg, model, data_sub):
        
    with torch.no_grad():
        bits, tot = 0.0, 0
        batch = [] # buffer, every time it fills up, we run it through the model

        for current in tqdm.trange(data_sub.size(0)):

            fr = max(0, current - arg.context)
            to = current + 1

            context = data_sub[fr:to].to(torch.long)
            if context.size(0) < arg.context + 1:
                pad = torch.zeros(size=(arg.context + 1 - context.size(0),), dtype=torch.long)
                context = torch.cat([pad, context], dim=0)

                assert context.size(0) == arg.context + 1

            if torch.cuda.is_available():
                context = context.cuda()

            batch.append(context[None, :])

            if len(batch) == arg.test_batchsize or current == data_sub.size(0) - 1:

                # batch is full, run it through the model
                b = len(batch)

                all = torch.cat(batch, dim=0)
                source = all[:, :-1] # input
                target = all[:, -1]  # target values

                output = model(source)

                lnprobs = output[torch.arange(b, device=d()), -1, target]
                log2probs = lnprobs * LOG2E # convert from nats to bits

                bits += - log2probs.sum()
                batch = [] # empty buffer

        bits_per_byte = bits / data_sub.size(0)
        
    return bits_per_byte

def finalize(arg, model, data_test):    
    
    model.load_state_dict(torch.load(os.path.join(arg.tb_dir, 'best_model.pt')))
    
    bits_per_byte = calculate_bpb(arg, model, data_test)    
    print(f'test data: {bits_per_byte:.4} bits per byte')
    
    generate_sequence(arg, model, data_test, starter=data_test[:arg.context], gensize=100000)

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # load the data
    arg.path = here('data') if arg.path is None else arg.path
    data_train, data_val, data_test = read_dataset(arg.path, arg.dataset)

    # create the model
    model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS, wide=arg.wide)
    
    if torch.cuda.is_available():
        model.cuda()
        
    print("Model parameters = %d" % sum(p.numel() for p in model.parameters()))

    if not arg.radam:
        opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())
        # linear learning rate warmup
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
    else:
        opt = RAdam(model.parameters(), lr=arg.lr)
    
    if USE_APEX:
        model, opt = amp.initialize(model, opt, opt_level="O1", verbosity=0)
    
    best_bpb = np.inf
    best_step = 0

    # training loop
    # - note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    for i in tqdm.trange(arg.num_batches):
    
        opt.zero_grad()

        # sample a batch of random subsequences
        starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - arg.context - 1)
        seqs_source = [data_train[start  :start+arg.context  ] for start in starts]
        seqs_target = [data_train[start+1:start+arg.context+1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
        # - target is the same sequence as source, except one character ahead

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()
        source, target = Variable(source), Variable(target)

        output = model(source)

        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        #tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, i * arg.batch_size)

        if not USE_APEX:
            loss.backward()
        else:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()        

        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()
        
        if not arg.radam:
            sch.step()
        
        # - validate every {arg.test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
            
            upto = arg.test_subset if arg.test_subset else data_val.size(0)            
            data_sub = data_val[:upto]
    
            bits_per_byte = calculate_bpb(arg, model, data_sub)

            # print validation performance. 1 bit per byte is (currently) state of the art.
            print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
            
            tag_scalar_dict = {'train-loss': float(loss.item()) * LOG2E, 'eval-loss': bits_per_byte}
            tbw.add_scalars(f'transformer/loss', tag_scalar_dict, i * arg.batch_size)                        

            if bits_per_byte < best_bpb:
                best_bpb = bits_per_byte
                best_step = i
                torch.save(model.state_dict(), os.path.join(arg.tb_dir, 'best_model.pt'))        
        
            print(f'best step {best_step}: {best_bpb:.4} bits per byte')            
            
            generate_sequence(arg, model, data_val)
            
    # load the best model, calculate bpb of the test data and generate some random text
    finalize(arg, model, data_test)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=50_000, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-D", "--path", dest="path",
                        help="Data path.",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0003, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=256, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=12, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=2500, type=int)
    
    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=None, type=int)    

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
                        default=512, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10000, type=int)
    
    parser.add_argument("--wide", dest="wide",
                        help="Use wide self attention instead of narrow self attention.",
                        action="store_true")
    
    parser.add_argument("--radam", dest="radam",
                        help="Use RAdam optimizer.",
                        action="store_true")
    
    parser.add_argument("--dataset",
                        dest="dataset",
                        choices=['phra_aphai', 'ramakien', 'samkok', 'mix'],
                        help="Selected dataset.",
                        default='phra_aphai')    

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
