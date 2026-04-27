import os
import sys
with open(sys.argv[0]) as f: code = f.read() # for logging, read the code of this file ASAP

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import glob
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import uuid
import time
import subprocess
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# NOTE: This is the AdamW-only baseline. The Muon optimizer is NOT used here.
# All parameters (transformer blocks + lm_head) are trained with AdamW.
# This is the comparison baseline for train_muon.py.
# -----------------------------------------------------------------------------

# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
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
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, bias = False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias = False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal = True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias = False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype = torch.long, device=idx.device)

        x = self.transformer.wte(idx)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("Error: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype = np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype = torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    # optimization hyperparams
    batch_size : int = 8*64          # batch size, in sequences, across all devices
    device_batch_size : int = 16     # batch size, in sequences, per device
    sequence_length : int = 1024     # sequence length, in tokens
    num_iterations : int = 6200      # number of iterations to run (full training)
    learning_rate : float = 0.0036
    warmup_iters : int = 200         # linear warmup iterations
    warmdown_iters : int = 1800      # linear warmdown iterations
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125       # how many steps between val loss evaluations
    val_tokens : int = 10485760      # validation tokens (fixed for consistent comparisons)
    save_every : int = 500           # save checkpoint every 500 steps for intermediate analysis
args = Hyperparameters()

# set up DDP (distributed data parallel)
assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)

# convenience variables
B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B*T*ddp_world_size) == 0
val_steps = args.val_tokens // (B*T*ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)


# init the optimizer — single AdamW for ALL parameters (AdamW-only baseline)
# In train_muon.py, the transformer blocks use Muon; here we replace that with AdamW.
optimizer1 = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
optimizers = [optimizer1]

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

train_losses = []
val_losses = []
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # evaluate validation loss
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

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
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            val_losses.append((step, val_loss.item()))
        torch.cuda.synchronize()
        t0 = time.time()

    # save checkpoint every save_every steps (and at the end)
    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000*(time.time() - t0)
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        x, y = train_loader.next_batch()

        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
    for p in model.parameters():
        p.grad /= train_accumulation_steps

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000*(time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        train_losses.append((step+1, train_loss.item()))

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# create loss plots
if master_process:
    steps_t, losses_t = zip(*train_losses)
    plt.figure()
    plt.plot(steps_t, losses_t)
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss (AdamW)')
    plt.savefig('logs/%s/train_loss.png' % run_id)

    steps_v, losses_v = zip(*val_losses)
    plt.figure()
    plt.plot(steps_v, losses_v)
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss (AdamW)')
    plt.savefig('logs/%s/val_loss.png' % run_id)

dist.destroy_process_group()
