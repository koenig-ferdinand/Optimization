"""
sweep_summary.py — Compare regularizer sweep results against V6 baselines
==========================================================================
Run after run_sequential.sh has finished:
    python "V6 - Train/slurm/sweep_summary.py"

Reads:
  • V6 - Train/log_muon.txt          — Muon baseline (target to beat)
  • V6 - Train/log_adamw.txt         — AdamW baseline (starting point)
  • V6 - Train/logs/{exp_name}/log.txt — one file per sweep experiment

Outputs (to slurm/results/):
  1. Console table  : final val_loss, Δ vs AdamW, Δ vs Muon, per experiment
  2. sweep_curves_<reg>.png   : val-loss curves per regularizer family
  3. sweep_ranking.png        : bar chart, all experiments ranked
  4. Best-λ table             : which λ won per regularizer
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V6_DIR     = os.path.dirname(SCRIPT_DIR)
LOG_ROOT   = os.path.join(V6_DIR, 'logs')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

MUON_LOG  = os.path.join(V6_DIR, 'log_muon.txt')
ADAMW_LOG = os.path.join(V6_DIR, 'log_adamw.txt')

# ── Experiment definitions (must match sweep_config.py) ───────────────────────

sys.path.insert(0, V6_DIR)
from sweep_config import EXPERIMENTS

REG_COLORS = {
    'none':           'black',
    'sv_variance':    '#1f77b4',
    'orthogonal':     '#ff7f0e',
    'effective_rank': '#2ca02c',
    'stable_rank':    '#d62728',
    'isometry':       '#9467bd',
    'dead_sv':        '#8c564b',
}


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_baseline_log(path):
    """Parse a V6 baseline log (log_muon.txt / log_adamw.txt).
    Returns {step: val_loss} dict."""
    data = {}
    if not os.path.exists(path):
        print(f'WARNING: baseline log not found: {path}')
        return data
    with open(path) as f:
        for line in f:
            m = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
            if m:
                data[int(m.group(1))] = float(m.group(2))
    return data


def parse_exp_log(log_path):
    """Parse a regularizer experiment log.
    Returns (steps, val_losses, early_stopped, missing)."""
    steps, losses = [], []
    early_stopped = False
    if not os.path.exists(log_path):
        return steps, losses, False, True   # missing
    with open(log_path) as f:
        for line in f:
            m = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            if 'EARLY_STOP' in line:
                early_stopped = True
    return steps, losses, early_stopped, False


# ── Load baselines ────────────────────────────────────────────────────────────

muon_data  = parse_baseline_log(MUON_LOG)
adamw_data = parse_baseline_log(ADAMW_LOG)

muon_steps  = sorted(muon_data.keys())
adamw_steps = sorted(adamw_data.keys())
muon_losses  = [muon_data[s]  for s in muon_steps]
adamw_losses = [adamw_data[s] for s in adamw_steps]

# Reference final losses — set to 3000 for Phase 1 sweep, 6200 for Phase 2 full runs
SWEEP_END   = 3000

# ── Same-schedule AdamW baseline (reg_name='none' experiment) ─────────────────
# If an experiment with reg_name='none' exists in EXPERIMENTS, use it as the
# AdamW comparison so the LR schedule matches the reg runs exactly.
# Falls back to log_adamw.txt if no such experiment exists.

_none_exp = next((e for e in EXPERIMENTS if e['reg_name'] == 'none'), None)
_none_log = os.path.join(LOG_ROOT, _none_exp['exp_name'], 'log.txt') if _none_exp else None
_none_steps, _none_losses, _, _none_missing = (
    parse_exp_log(_none_log) if _none_log else ([], [], False, True)
)
_none_lut = dict(zip(_none_steps, _none_losses))

# Use same-schedule baseline if available and complete, else fall back
if _none_losses and not _none_missing:
    adamw_same  = _none_lut
    adamw_label = f'AdamW ({_none_exp["exp_name"]}, same schedule)'
    print(f'[INFO] Using same-schedule AdamW baseline: {_none_exp["exp_name"]}')
else:
    adamw_same  = dict(zip(adamw_steps, adamw_losses))
    adamw_label = 'AdamW (log_adamw.txt, 6200-step schedule)'
    print(f'[INFO] No same-schedule AdamW found — using log_adamw.txt')

muon_final  = muon_data.get(SWEEP_END,  muon_losses[-1])
adamw_final = adamw_same.get(SWEEP_END, _none_losses[-1] if _none_losses else adamw_losses[-1])
muon_gap    = adamw_final - muon_final


# ── Load experiment results ───────────────────────────────────────────────────

results = []
for exp in EXPERIMENTS:
    name     = exp['exp_name']
    log_path = os.path.join(LOG_ROOT, name, 'log.txt')
    steps, losses, early_stopped, missing = parse_exp_log(log_path)

    fl    = losses[-1] if losses else float('nan')
    ls    = steps[-1]  if steps  else 0
    d_adamw = fl - adamw_final if not np.isnan(fl) else float('nan')
    d_muon  = fl - muon_final  if not np.isnan(fl) else float('nan')
    gap_closed = (d_adamw / muon_gap * -100) if (not np.isnan(d_adamw) and muon_gap > 0) else float('nan')

    results.append({
        'exp_name':    name,
        'reg_name':    exp['reg_name'],
        'lam':         exp['reg_lambda'],
        'steps':       steps,
        'losses':      losses,
        'final_loss':  fl,
        'last_step':   ls,
        'd_adamw':     d_adamw,   # negative = better than AdamW (same-schedule)
        'd_muon':      d_muon,    # negative = better than Muon
        'gap_closed':  gap_closed,
        'early_stopped': early_stopped,
        'missing':     missing,
    })


# ── Console table ─────────────────────────────────────────────────────────────

COL = 34
print()
print('=' * 95)
print(f'  Baselines at step {SWEEP_END}:   '
      f'AdamW = {adamw_final:.4f}   '
      f'Muon = {muon_final:.4f}   '
      f'Gap = {muon_gap:.4f}')
print('=' * 95)
print(f'{"Experiment":<{COL}} {"Reg":<16} {"λ":>7}  {"Steps":>5}  '
      f'{"ValLoss":>7}  {"ΔAdamW":>7}  {"ΔMuon":>7}  {"Gap%":>6}  Status')
print('-' * 95)

for r in sorted(results, key=lambda x: x['final_loss']):
    status = ('MISSING'    if r['missing']
              else 'STOPPED' if r['early_stopped']
              else 'OK')
    fl_s  = f'{r["final_loss"]:.4f}' if not np.isnan(r['final_loss']) else '  —'
    da_s  = f'{r["d_adamw"]:+.4f}'   if not np.isnan(r['d_adamw'])   else '  —'
    dm_s  = f'{r["d_muon"]:+.4f}'    if not np.isnan(r['d_muon'])    else '  —'
    gc_s  = f'{r["gap_closed"]:+.1f}%' if not np.isnan(r['gap_closed']) else '  —'
    print(f'{r["exp_name"]:<{COL}} {r["reg_name"]:<16} {r["lam"]:>7.0e}  '
          f'{r["last_step"]:>5}  {fl_s:>7}  {da_s:>7}  {dm_s:>7}  {gc_s:>6}  {status}')

print('-' * 95)
print(f'{"AdamW baseline (same schedule)":<{COL}} {"—":<16} {"—":>7}  {SWEEP_END:>5}  '
      f'{adamw_final:>7.4f}  {"0.0000":>7}  {(-muon_gap):>7.4f}  {"0.0%":>6}  REF')
print(f'{"Muon baseline (target)":<{COL}} {"—":<16} {"—":>7}  {SWEEP_END:>5}  '
      f'{muon_final:>7.4f}  {(-muon_gap):>7.4f}  {"0.0000":>7}  {"100.0%":>6}  TARGET')
print('=' * 95)


# ── Detailed trajectory (every 100 steps) ────────────────────────────────────

# Lookups: {step: loss}
_traj     = {r['exp_name']: dict(zip(r['steps'], r['losses'])) for r in results}
_adamw_lut = adamw_same   # same-schedule baseline if available, else log_adamw.txt
_muon_lut  = dict(zip(muon_steps,  muon_losses))

# All 100-step checkpoints that appear in any experiment's log
_all_steps = sorted({s for r in results for s in r['steps'] if s % 100 == 0})
if not _all_steps:
    _all_steps = list(range(100, SWEEP_END + 1, 100))

_PER_ROW = 9   # step:loss tokens per printed line

def _traj_rows(lut, label, adamw_lut, steps):
    """Yield rows of formatted step:loss tokens for one series."""
    tokens = []
    for s in steps:
        v = lut.get(s)
        if v is None:
            break                          # series ended early — stop here
        av = adamw_lut.get(s, float('nan'))
        marker = '↓' if not np.isnan(av) and v < av else ' '
        tokens.append(f'{s:>4}:{v:.4f}{marker}')
    return tokens

print()
print('─' * 95)
print('  Val-loss trajectory  (every 100 steps)   ↓ = below AdamW at that step')
print('─' * 95)

# ── Print AdamW & Muon reference rows once ───────────────────────────────────
for ref_label, ref_lut in [('  AdamW ref', _adamw_lut), ('   Muon ref', _muon_lut)]:
    ref_tokens = []
    for s in _all_steps:
        v = ref_lut.get(s)
        ref_tokens.append(f'{s:>4}:{v:.4f} ' if v is not None else f'{s:>4}:  —    ')
    if ref_tokens:
        print(f'\n{ref_label}  ·  baseline')
        for i in range(0, len(ref_tokens), _PER_ROW):
            print('    ' + '  '.join(ref_tokens[i:i + _PER_ROW]))

print()
print('─' * 95)

# ── Print one block per experiment, grouped by regularizer ───────────────────
cur_reg = None
for r in results:          # keep original EXPERIMENTS order
    lut = _traj[r['exp_name']]

    # Separator between regularizer families
    if r['reg_name'] != cur_reg:
        cur_reg = r['reg_name']
        print(f'\n  ── {cur_reg} ──')

    status_tag = '  [STOPPED]' if r['early_stopped'] else ('  [MISSING]' if r['missing'] else '')
    print(f'\n  {r["exp_name"]}  ·  λ={r["lam"]:.0e}{status_tag}')

    tokens = _traj_rows(lut, r['exp_name'], _adamw_lut, _all_steps)
    if not tokens:
        print('    (no data)')
        continue
    for i in range(0, len(tokens), _PER_ROW):
        print('    ' + '  '.join(tokens[i:i + _PER_ROW]))

print()
print('─' * 95)


# ── Best-λ per regularizer ────────────────────────────────────────────────────

print('\nBest λ per regularizer (lowest val_loss at sweep end):')
print('-' * 70)
reg_groups = {}
for r in results:
    reg_groups.setdefault(r['reg_name'], []).append(r)

for reg_name, runs in reg_groups.items():
    valid = [r for r in runs if not np.isnan(r['final_loss']) and not r['missing']]
    if not valid:
        print(f'  {reg_name:<20}  — all missing / stopped early')
        continue
    best = min(valid, key=lambda x: x['final_loss'])
    gc   = f'{best["gap_closed"]:+.1f}%' if not np.isnan(best['gap_closed']) else '—'
    print(f'  {reg_name:<20}  best λ = {best["lam"]:.0e}   '
          f'val_loss = {best["final_loss"]:.4f}  '
          f'ΔAdamW = {best["d_adamw"]:+.4f}  '
          f'gap closed = {gc}')
print()


# ── Plot 1: val-loss curves per regularizer family ───────────────────────────

all_regs = [r for r in REG_COLORS if r != 'none']
ncols = 3
nrows = int(np.ceil(len(all_regs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(17, 5 * nrows), dpi=150, sharey=True)
axes = axes.flatten()

for ax_i, reg_name in enumerate(all_regs):
    ax = axes[ax_i]

    # Baselines
    ax.plot(adamw_steps, adamw_losses, color='#C44E52', linewidth=2.2,
            linestyle='--', label=f'AdamW baseline ({adamw_final:.4f})', zorder=6)
    ax.plot(muon_steps,  muon_losses,  color='#4C72B0', linewidth=2.2,
            linestyle='--', label=f'Muon baseline ({muon_final:.4f})',  zorder=6)

    # Experiments for this regularizer
    runs = sorted([r for r in results if r['reg_name'] == reg_name], key=lambda x: x['lam'])
    n    = max(len(runs), 1)
    cols = cm.Oranges(np.linspace(0.45, 0.95, n))

    for run, col in zip(runs, cols):
        if not run['steps']:
            continue
        style = ':' if run['early_stopped'] else '-'
        lbl   = f'λ={run["lam"]:.0e}  →  {run["final_loss"]:.4f}'
        if run['early_stopped']:
            lbl += ' (stopped)'
        ax.plot(run['steps'], run['losses'], color=col, linewidth=1.6,
                linestyle=style, label=lbl)

    ax.set_title(reg_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xlabel('Iteration', fontsize=8)
    ax.set_ylabel('Val Loss', fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, SWEEP_END + 100)
    ax.tick_params(labelsize=7)

    # Shade the Muon–AdamW gap at sweep end
    ax.axhspan(muon_final, adamw_final, color='grey', alpha=0.08,
               label='_nolegend_')

# Hide unused axes
for i in range(len(all_regs), len(axes)):
    axes[i].set_visible(False)

fig.suptitle(
    f'Sweep — Val Loss vs Muon & AdamW Baselines  (step {SWEEP_END})\n'
    f'Muon gap to close: {muon_gap:.4f}  |  AdamW={adamw_final:.4f}  Muon={muon_final:.4f}',
    fontsize=12,
)
fig.tight_layout()
out = os.path.join(OUT_DIR, 'sweep_curves.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')


# ── Plot 2: ranked bar chart ──────────────────────────────────────────────────

valid = [r for r in results if not np.isnan(r['final_loss']) and not r['missing']]
valid.sort(key=lambda x: x['final_loss'])

fig, ax = plt.subplots(figsize=(12, max(5, len(valid) * 0.45 + 1.5)), dpi=150)
labels = [f'{r["reg_name"]}  λ={r["lam"]:.0e}' for r in valid]
values = [r['final_loss'] for r in valid]
colors = [REG_COLORS.get(r['reg_name'], 'gray') for r in valid]

bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(adamw_final, color='#C44E52', linewidth=2.0, linestyle='--',
           label=f'AdamW baseline  {adamw_final:.4f}')
ax.axvline(muon_final,  color='#4C72B0', linewidth=2.0, linestyle='--',
           label=f'Muon baseline   {muon_final:.4f}')
ax.axvspan(muon_final, adamw_final, color='grey', alpha=0.10, label='Muon–AdamW gap')

ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=8)
ax.set_xlabel(f'Val Loss at Step {SWEEP_END}  (lower = better)', fontsize=10)
ax.set_title('Sweep Ranking — All Regularizer Experiments vs Baselines', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, axis='x', alpha=0.25)
lo = min(values + [muon_final]) - 0.02
hi = max(values + [adamw_final]) + 0.06
ax.set_xlim(lo, hi)
ax.invert_yaxis()
fig.tight_layout()
out = os.path.join(OUT_DIR, 'sweep_ranking.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')
print('\nDone. Update sweep_config.py Phase 2 lambdas, then resubmit for full runs.')
