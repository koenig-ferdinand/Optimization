import os
import sys
with open(sys.argv[0]) as f: code = f.read()  # log source code

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
import argparse
import matplotlib.pyplot as plt

# =============================================================================
# Muon optimizer  (identical to train_muon_free.py)
# =============================================================================

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                # attn.c_attn is (2304, 768) = 3 × 768 — split Q/K/V
                if g.size(0) == 3 * g.size(1):
                    scale = g.size(1) ** 0.5
                    q, k, v = g.split(g.size(1))
                    g = torch.cat([zeropower_backend(q, steps=group['backend_steps']),
                                   zeropower_backend(k, steps=group['backend_steps']),
                                   zeropower_backend(v, steps=group['backend_steps'])])
                else:
                    scale = max(g.size(0), g.size(1)) ** 0.5
                    g = zeropower_backend(g, steps=group['backend_steps'])
                p.data.add_(g, alpha=-lr * scale)


# =============================================================================
# GPT-2 model  (identical to train_muon_free.py)
# =============================================================================

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
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
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head    = config.n_head
        self.n_embd    = config.n_embd
        self.head_dim  = self.n_embd // self.n_head
        self.split_qkv = config.split_qkv
        if config.split_qkv:
            # Separate projections so Q, K can stay AdamW while V goes to Muon
            self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        else:
            self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary  = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        if self.split_qkv:
            q = self.c_q(x)
            k = self.c_k(x)
            v = self.c_v(x)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer) ** 0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    vocab_size : int  = 50257
    n_layer    : int  = 12
    n_head     : int  = 12
    n_embd     : int  = 768
    split_qkv  : bool = False   # use separate c_q/c_k/c_v instead of fused c_attn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)
        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss


# =============================================================================
# Data loader  (identical to train_muon_free.py)
# =============================================================================

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    assert header[0] == 20240520
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B, self.T = B, T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0
        self.ntok_total = sum(_peek_data_shard(f) for f in self.files)
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[0])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()


# =============================================================================
# Hyperparameters + CLI
# =============================================================================

@dataclass
class Hyperparameters:
    input_bin     : str   = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str   = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size         : int   = 8 * 64
    device_batch_size  : int   = 16
    sequence_length    : int   = 1024
    num_iterations     : int   = 3000
    learning_rate      : float = 0.0036
    warmup_iters       : int   = 0
    warmdown_iters     : int   = 870       # proportional 3000-step warmdown
    weight_decay       : float = 0.0
    val_loss_every     : int   = 100
    val_tokens         : int   = 10485760
    save_every         : int   = 100
    exp_name           : str   = ''
    # ── Hybrid control ──────────────────────────────────────────────────────────
    # muon_layers:       'all' or comma-separated layer indices, e.g. '0,1,2'
    # muon_matrices:     'all' or comma-separated sub-paths,  e.g. 'mlp.c_fc'
    #   split_qkv=False: mlp.c_fc  mlp.c_proj  attn.c_attn  attn.c_proj
    #   split_qkv=True:  mlp.c_fc  mlp.c_proj  attn.c_q  attn.c_k  attn.c_v  attn.c_proj
    # muon_lr_ratio:     Muon LR = muon_lr_ratio × AdamW LR  (attention matrices)
    # muon_mlp_lr_ratio: override LR ratio for MLP matrices; 0 = use muon_lr_ratio
    # split_qkv:         split c_attn into separate c_q/c_k/c_v projections
    muon_layers       : str   = 'all'
    muon_matrices     : str   = 'all'
    muon_lr_ratio     : float = 0.1
    muon_mlp_lr_ratio : float = 0.0
    split_qkv         : bool  = False

args = Hyperparameters()

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument('--exp_name',           type=str,   default=args.exp_name)
_p.add_argument('--num_iterations',     type=int,   default=args.num_iterations)
_p.add_argument('--warmdown_iters',     type=int,   default=args.warmdown_iters)
_p.add_argument('--save_every',         type=int,   default=args.save_every)
_p.add_argument('--muon_layers',        type=str,   default=args.muon_layers,
                help="'all' or comma-separated layer indices, e.g. '0,1,2'")
_p.add_argument('--muon_matrices',      type=str,   default=args.muon_matrices,
                help="'all' or comma-separated matrix paths, e.g. 'mlp.c_fc,mlp.c_proj'")
_p.add_argument('--muon_lr_ratio',      type=float, default=args.muon_lr_ratio,
                help='Muon LR = muon_lr_ratio × AdamW LR (applies to attention matrices)')
_p.add_argument('--muon_mlp_lr_ratio',  type=float, default=args.muon_mlp_lr_ratio,
                help='Override Muon LR ratio for MLP matrices; 0 = use muon_lr_ratio')
_p.add_argument('--split_qkv',          action='store_true', default=args.split_qkv,
                help='Use separate c_q/c_k/c_v projections instead of fused c_attn')
_cli, _ = _p.parse_known_args()
args.exp_name           = _cli.exp_name
args.num_iterations     = _cli.num_iterations
args.warmdown_iters     = _cli.warmdown_iters
args.save_every         = _cli.save_every
args.muon_layers        = _cli.muon_layers
args.muon_matrices      = _cli.muon_matrices
args.muon_lr_ratio      = _cli.muon_lr_ratio
args.muon_mlp_lr_ratio  = _cli.muon_mlp_lr_ratio
args.split_qkv          = _cli.split_qkv

# Parse muon_layers / muon_matrices into usable sets
_muon_layer_ids  = None if args.muon_layers == 'all' \
                   else set(int(x) for x in args.muon_layers.split(','))
# When split_qkv=True the fused c_attn is replaced by c_q/c_k/c_v, so 'all'
# expands to the larger set of 6 individual projections instead of 4.
_ALL_MATRICES    = ({'mlp.c_fc', 'mlp.c_proj',
                     'attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj'}
                    if args.split_qkv else
                    {'mlp.c_fc', 'mlp.c_proj', 'attn.c_attn', 'attn.c_proj'})
_muon_matrix_set = _ALL_MATRICES if args.muon_matrices == 'all' \
                   else set(m.strip() for m in args.muon_matrices.split(','))


# =============================================================================
# DDP setup
# =============================================================================

assert torch.cuda.is_available()
dist.init_process_group(backend='nccl')
ddp_rank       = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device         = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process  = (ddp_rank == 0)

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

train_loader = DistributedDataLoader(args.input_bin,     B, T, ddp_rank, ddp_world_size)
val_loader   = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: {train_loader.ntok_total} tokens across {len(train_loader.files)} files")
    print(f"Validation DataLoader: {val_loader.ntok_total} tokens across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# =============================================================================
# Model
# =============================================================================

model = GPT(GPTConfig(split_qkv=args.split_qkv)).cuda()
model = torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# =============================================================================
# Optimizer setup — the key part of this script
# =============================================================================
# We split parameters into two groups:
#   muon_params  → Muon optimizer  (selected layers × selected matrices, 2D only)
#   adamw_params → AdamW optimizer (everything else: embeddings, lm_head,
#                                   non-selected layers, 1D params)
#
# Rule: a parameter gets Muon if AND ONLY IF:
#   1. Its layer index is in muon_layer_ids (or muon_layers='all')
#   2. Its matrix path is in muon_matrix_set (or muon_matrices='all')
#   3. It is 2-dimensional (Muon only works on matrices)

# muon_attn_params: attention weight matrices → standard Muon LR (muon_lr_ratio)
# muon_mlp_params:  MLP weight matrices      → potentially lower LR (muon_mlp_lr_ratio)
# adamw_params:     everything else (embeddings, lm_head, biases, non-selected params)
muon_attn_params = []
muon_mlp_params  = []
adamw_params     = []

n_layers   = len(raw_model.transformer.h)
muon_count = 0

for layer_idx, block in enumerate(raw_model.transformer.h):
    layer_in_muon = (_muon_layer_ids is None) or (layer_idx in _muon_layer_ids)
    for param_path, param in block.named_parameters():
        matrix_path    = '.'.join(param_path.split('.')[:2])   # e.g. 'mlp.c_fc'
        matrix_in_muon = matrix_path in _muon_matrix_set
        is_mlp         = param_path.startswith('mlp.')
        if layer_in_muon and matrix_in_muon and param.dim() == 2:
            (muon_mlp_params if is_mlp else muon_attn_params).append(param)
            muon_count += 1
        else:
            adamw_params.append(param)

# Embeddings and lm_head always go to AdamW
for param in raw_model.transformer.wte.parameters():
    adamw_params.append(param)
for param in raw_model.lm_head.parameters():
    # weight-tied with wte — only add if not already added
    if not any(param is p for p in adamw_params):
        adamw_params.append(param)

_muon_attn_lr = args.muon_lr_ratio * args.learning_rate
_muon_mlp_lr  = (args.muon_mlp_lr_ratio * args.learning_rate
                 if args.muon_mlp_lr_ratio > 0.0 else _muon_attn_lr)

if master_process:
    total_params     = sum(p.numel() for p in raw_model.parameters())
    muon_attn_count  = sum(p.numel() for p in muon_attn_params)
    muon_mlp_count   = sum(p.numel() for p in muon_mlp_params)
    muon_param_count = muon_attn_count + muon_mlp_count
    print(f'{"="*60}')
    print(f'Hybrid optimizer split:')
    print(f'  muon_layers       : {args.muon_layers}')
    print(f'  muon_matrices     : {args.muon_matrices}')
    print(f'  split_qkv         : {args.split_qkv}')
    print(f'  Muon attn params  : {muon_attn_count:,}  lr={_muon_attn_lr:.2e}')
    print(f'  Muon MLP  params  : {muon_mlp_count:,}  lr={_muon_mlp_lr:.2e}')
    print(f'  Muon total        : {muon_param_count:,} / {total_params:,} '
          f'({100*muon_param_count/total_params:.1f}%)  [{muon_count} matrices]')
    print(f'  AdamW params      : {total_params - muon_param_count:,}  lr={args.learning_rate:.2e}')
    print(f'{"="*60}')

optimizer_adamw = torch.optim.AdamW(
    adamw_params, lr=args.learning_rate, betas=(0.9, 0.95),
    weight_decay=args.weight_decay, fused=True)
optimizers = [optimizer_adamw]

# Use a single Muon optimizer when LRs are identical (backward compat),
# or two separate Muon optimizers when MLP needs a different LR.
if args.muon_mlp_lr_ratio > 0.0:
    if muon_attn_params:
        optimizers.append(Muon(muon_attn_params, lr=_muon_attn_lr, momentum=0.95))
    if muon_mlp_params:
        optimizers.append(Muon(muon_mlp_params,  lr=_muon_mlp_lr,  momentum=0.95))
else:
    muon_all_params = muon_attn_params + muon_mlp_params
    if muon_all_params:
        optimizers.append(Muon(muon_all_params, lr=_muon_attn_lr, momentum=0.95))

# =============================================================================
# LR schedule  (same shape as muon_3000 / adamw_3000)
# =============================================================================

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# =============================================================================
# Logging
# =============================================================================

_V6_DIR = os.path.dirname(os.path.abspath(__file__))
if master_process:
    if args.exp_name:
        logdir  = os.path.join(_V6_DIR, 'logs', args.exp_name)
        logfile = os.path.join(logdir, 'log.txt')
    else:
        run_id  = str(uuid.uuid4())
        logdir  = os.path.join(_V6_DIR, 'logs', run_id)
        logfile = os.path.join(_V6_DIR, 'logs', f'{run_id}.txt')
    os.makedirs(logdir, exist_ok=True)
    with open(logfile, 'w') as f:
        f.write('=' * 100 + '\n')
        f.write(f'exp_name: {args.exp_name} | '
                f'muon_layers: {args.muon_layers} | '
                f'muon_matrices: {args.muon_matrices} | '
                f'muon_lr_ratio: {args.muon_lr_ratio} | '
                f'muon_mlp_lr_ratio: {args.muon_mlp_lr_ratio} | '
                f'split_qkv: {args.split_qkv}\n')
        f.write('=' * 100 + '\n')
        f.write(code)
        f.write('=' * 100 + '\n')
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('=' * 100 + '\n')

# =============================================================================
# Training loop  (identical structure to train_muon_free.py)
# =============================================================================

train_losses = []
val_losses   = []
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # ── Validation ────────────────────────────────────────────────────────────
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = torch.tensor(0.0, device=device)
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} '
                  f'train_time:{training_time_ms:.0f}ms '
                  f'step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, 'a') as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} '
                        f'train_time:{training_time_ms:.0f}ms '
                        f'step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            val_losses.append((step, val_loss.item()))
        torch.cuda.synchronize()
        t0 = time.time()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        torch.save({'step': step, 'model': raw_model.state_dict()},
                   os.path.join(logdir, f'state_step{step:06d}.pt'))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # ── Training step ─────────────────────────────────────────────────────────
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

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f'step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} '
              f'train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms')
        with open(logfile, 'a') as f:
            f.write(f'step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} '
                    f'train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n')
        train_losses.append((step + 1, train_loss.item()))

if master_process:
    print(f'peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB')

# ── Plots ─────────────────────────────────────────────────────────────────────
if master_process and train_losses:
    steps_t, losses_t = zip(*train_losses)
    plt.figure(); plt.plot(steps_t, losses_t)
    plt.xlabel('Step'); plt.ylabel('Training Loss')
    plt.title(f'Train Loss — {args.exp_name}')
    plt.savefig(os.path.join(logdir, 'train_loss.png'))

    steps_v, losses_v = zip(*val_losses)
    plt.figure(); plt.plot(steps_v, losses_v)
    plt.xlabel('Step'); plt.ylabel('Validation Loss')
    plt.title(f'Val Loss — {args.exp_name}')
    plt.savefig(os.path.join(logdir, 'val_loss.png'))
