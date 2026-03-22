import os #  standard Python library for interacting with the operating system. 
import sys
with open(sys.argv[0]) as f: code = f.read() # for logging, read the code of this file ASAP

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import glob
import torch.distributed as dist # PyTorch's multi-GPU communication library
import torch._inductor.config as config # This is the configuration for torch.compile's backend
from torch.nn.parallel import DistributedDataParallel as DDP # DDP wrapper, which takes a model and automatically splits work across multiple GPUs
import uuid
import time
import subprocess
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Muon optimizer

# orthogonal matrix stripped of singular values
def zeropower_via_svd(G, steps = None): 
    U, S, V = G.svd()
    return U @ V.T

# approximate nearest orthogonal matrix: X = U @ diag(aσ + bσ³ + cσ⁵) @ V.T
def zeropower_via_newtonschulz5(G, steps = 5, eps = 1e-7): 
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()/ (G.norm() + eps) # normalize: highest singular value <= 1

    if G.size(0) > G.size(1): # for numerical optimization
        X = X.T 

    for _ in range(steps): 
        A = X @ X.T
        B = A @ X
        X = a*X + b*B + c*A@B
    
    if G.size(0) > G.size(1): # return to original shape
        X = X.T 

    return X.to(G.dtype)

zeropower_backends = dict(svd = zeropower_via_svd, newtonschulz5 = zeropower_via_newtonschulz5)

#define muon optimizer as class, inheriting the torch.optim.Optimizer class 
#=> Muon takes all methods and attributes of torch.optim.Optimizer class: param_groups, state, zero_grad, ...
class Muon(torch.optim.Optimizer): 
    #constructor
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self): 
        for group in self.param_groups: 
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            for p in group['params']: 
                #get the gradient for each weight matrix
                g = p.grad
                if g is None: 
                    continue 
                state = self.state[p]
                if 'momentum_buffer' not in state: 
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g) # buf = 0.95 * buf + gradient
                if group['nesterov']: 
                    g = g.add(buf, alpha=momentum) # g = g + momentum * buf
                # transformer attention layer Q, K, V are combined into one big (2304, 768) matrix
                # --> split into 3 weight matrices and orthogonalize separately 
                if g.size(0) == 3 * g.size(1): 
                    q_grad, k_grad, v_grad = g.split(g.size(1))
                    q_orth = zeropower_backend(q_grad, steps=group['backend_steps'])
                    k_orth = zeropower_backend(k_grad, steps=group['backend_steps'])
                    v_orth = zeropower_backend(v_grad, steps=group['backend_steps'])
                    g = torch.cat([q_orth, k_orth, v_orth])
                    scale = g.size(1)**0.5
                else: 
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5

                p.data.add_(g, alpha=-lr*scale) # actual weight updatate: weights = weights - lr*scale * orthoganolied_gradient

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

# Rotary class to rotate Q, K for position awarness 
class Rotary(torch.nn.Module): 

    def __init__(self, dim, base = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None 

    # rotate Q or K tensor to encode position information
    # Q and K are of the form: (B, T, n_head, head_dim) (multihead attention)
    def forward(self, x): 
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) # [0, 1, ..., T-1]
            freqs = torch.outer(t, self.inv_freq).to(x.device) # (T, 32) matrix
            self.cos_cached = freqs.cos() # (T, 32) matrix
            self.sin_cached = freqs.sin() # (T, 32) matrix
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
    
def apply_rotary_emb(x, cos, sin): 
    assert x.ndim == 4 #multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps = 1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0 
        # key, query, value projections for all heads, in a batch
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, bias = False) # linear layer combining Q, K, V
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)  # linear layer 768×768 weight matrix
        self.rotary = Rotary(self.head_dim)

    def forward(self, x): 
        B, T, C = x.size() # batch size (4), sequence length (each T = 1024 tokens ), embedding dimensinoality (n_embd) (768 dim vector)
        qkv = self.c_attn(x) # Q, K, V all at once:  # (B, T, 768) → (B, T, 2304)
        q, k, v = qkv.split(self.n_embd, dim = 2) # each (B, T, 768)
        q = q.view(B, T, self.n_head, self.head_dim) # Split 768 into 12 heads × 64 dims (B, T, 12, 64)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin) # inject position into Q, K 
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), # (B, 12, T, 64)
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            is_causal = True # each token can only attend to previous tokens
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # back to (B, T, 768)
        # output projection: 12 heads independently computed 64 dim output, concatenated them back, now allow them to talk to each other: 
        y = self.c_proj(y) # takes the 768-dim vector,  multiplies it by weight matrix , gives another 768-dim vector
        return y
    
class MLP(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = False) # weight matrix expanding vector from 768 → 3072
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = False) # weight matrix compressing vector 3072 → 768

    def forward(self, x): 
        x = self.c_fc(x) # (B, T, 768) → (B, T, 3072)
        x = F.gelu(x) # nonlinear activation function: GELU is a smooth version of ReLU 
        x = self.c_proj(x)  # (B, T, 3072) → (B, T, 768)
        return x


# transformer block, the model stacks 12 of these 
class Block(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        # two components: attention and mlp
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)
    
    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass # @dataclass auto-generates the constructor.
class GPTConfig: 
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module): 

    def __init__(self, config): 
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # wte is the embedding table a 50257×768 matrix: each token gets mapped to 768 dim vector
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # list of 12 Blocks
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # output layer: takes 768 vector to produce 50257 dim probability distribution
        self.transformer.wte.weight = self.lm_head.weight # embedding matrix and output layer are literally the same: one maps token→vector, the other maps vector→token 

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size() # tensor of lists of token ids of shape (B,T) 
        pos = torch.arange(0, t, dtype = torch.long, device=idx.device) # position indeces (actually unused)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # (B, T) → (B, T, 768) look up each token id, now each token is a 768 dim vector

        for block in self.transformer.h: # pass through all transformer blocks
            x = block(x)
        x = rmsnorm(x) # final normalization

        # two cases: training (targets provided), inference (no targets)
        if targets is not None: 
            logits = self.lm_head(x)   # (B, T, 768) → (B, T, 50257)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # compares scores against actual token 
        else: 
            logits = self.lm_head(x[:, [-1], :]) # only last token
            logits = logits.float()
            loss = None
        
        if not return_logits: 
            logits = None
        
        return logits, loss
    
# -----------------------------------------------------------------------------
# Simple Distributed Data Loader
# data format simple: a .bin file containing 1024-byte header (magic numer, version, token count, padding), a flat sequence of token IDs as uint16
# use same format but much smaller dataset (FineWeb is 10billion tokens, consider TinyShakespear )
# Keller Jordan's repo has dev/data/tinyshakespeare.py that downloads Shakespeare text, tokenizes it, and writes .bin files with this exact header structure. 

# only takes a look at the data 
def _peek_data_shard(filename): 
    # only reads the header, returns header data
    with open(filename, "rb") as f: 
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520: 
        print("Error: magic numer mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok # for now just return the number of tokens

# actually loads the data 
def _load_data_shard(filename): 
    with open(filename, "rb") as f: 
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype = np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens 

class DistributedDataLoader: 
    def __init__(self, filename_pattern, B, T, process_rank, num_processes): 
        self.process_rank = process_rank # which GPU am I? (0 for single device)
        self.num_processes = num_processes # how many GPUs total? (1 for single device)
        self.B = B  # batch size
        self.T = T # sequence length

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern)) # find all data files matching the pattern
        assert len(self.files) > 0, f"did not find any files that mathc the pattern {filename_pattern}"
        
        # load any validate all data sharts, count number of tokens in total 
        ntok_total = 0
        for fname in self.files:   #loop through every file
            shard_ntok = _peek_data_shard(fname) # peak at header to get token count
            assert shard_ntok >= num_processes * B * T + 1 # Assert each file is big enough to produce at least one batch for all GPUs
            ntok_total  += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self): 
        self.current_shard = 0 # start at first file 
        self.current_position = self.process_rank * self.B * self.T # each GPU starts at different offset
        self.tokens = _load_data_shard(self.files[self.current_shard]) # actauly load the first tile's tokens into memory

    def advance(self): 
        self.current_shard = (self.current_shard + 1) % len(self.files) # Move to the next file.
        self.current_position = self.process_rank * self.B * self.T # Reset read position for the new file.
        self.tokens = _load_data_shard(self.files[self.current_shard]) # Load the new file.

    def next_batch(self): 
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] # grab slice of tokens. Why B*T+1? Because you need inputs AND targets.
        buf = torch.tensor(buf.astype(np.int32), dtype = torch.long) # convert tino numpy uint16 pytorch tensor of intgers
        # x, y are offset by one — so if x has tokens at positions [0,1,2,3], y has [1,2,3,4]. 
        x = (buf[:-1]).view(B, T) # inputs (everything except the last)
        y = (buf[1:]).view(B, T) # targets (everything except the first)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens): 
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass 
class Hyperparameters: 
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 2*32         #ORIGINAL: 8*64   # batch size, in sequences, across all devices
    device_batch_size : int = 32    #ORIGINAL: 64    # batch size, in sequences, per device
    sequence_length : int = 256     #ORIGINAL: 1024  # sequence length, in tokens
    num_iterations : int = 100      #ORIGINAL: 6200  # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 30       #ORIGINAL: 1800  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 10       #ORIGINAL: 125   # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 16384       #ORIGINAL:10485760   # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# set up DDP (distributed data parallel)
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device:{device}")
master_process = (ddp_rank == 0)  # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate number of steps to take in the val loop
assert args.val_tokens % (B*T*ddp_world_size) == 0
val_steps = args.val_tokens // (B*T*ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desirged global batch size
assert args.batch_size % (B * ddp_world_size) == 0 
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens 
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process: 
    print(f"Training DataLoader: total number of tokens:{train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens:{val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"): 
    config.coordinate_descent_tuning = True
model = torch.compile(model)
model = DDP(model, device_ids =[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)


# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]

#learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it): 
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters: 
        return 1.0
    # 3) lineaer warmdown
    else: 
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

#begin logging
if master_process: 
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__} compile for CUDA {torch.version.cuda} \n nvidia-smi \n")
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 +'\n')

train_losses = []
val_losses = []
training_time_ms = 0
# start the cuda clock
torch.cuda.synchronize()
t0=time.time()

# begin training

train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # following Jordan we ignore the first 10
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1  # <= 11 to avoid bug in val


    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)): 
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0

        for _ in range(val_steps): 
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): 
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
    
        if master_process: 
            print(f'step: {step}/{args.num_iterations} val_loss: {val_loss:.4f} train_time: {training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f: 
                f.write(f'step: {step}/{args.num_iterations} val_loss: {val_loss:.4f} train_time: {training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms \n')
            val_losses.append((step, val_loss.item()))
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop clock
        torch.cuda.synchronize()
        training_time_ms += 1000*(time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers = [opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        #start clock again
        torch.cuda.synchronize()
        t0 = time.time()


    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step: 
        break
    
    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1): 
        with ctx: 
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for next batch
        x, y = train_loader.next_batch()

        # backward pass
        if i < train_accumulation_steps: 
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else: 
            loss.backward() # just sync on the last step
    for p in model.parameters(): 
        p.grad /= train_accumulation_steps
    
    # step the optimizer and schedulers
    for opt, sched in zip(optimizers, schedulers): 
        opt.step()
        sched.step()
    # null the gradients 
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.
    if master_process: 
        approx_time = training_time_ms + 1000*(time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time: {approx_time:.0f} step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f: 
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time: {approx_time:.0f} step_avg:{approx_time/timed_steps:.2f}ms\n")
        train_losses.append((step+1, train_loss.item()))

if master_process: 
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# create plot
if master_process: 
    steps_t, losses_t = zip(*train_losses)
    plt.figure()
    plt.plot(steps_t, losses_t)
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.savefig('logs/%s/train_loss.png' %run_id)

    steps_v, losses_v = zip(*val_losses)
    plt.figure()
    plt.plot(steps_v, losses_v)
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.savefig('logs/%s/val_loss.png' %run_id)


# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()

