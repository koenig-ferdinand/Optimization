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

# ── Tee stdout → console + results/sweep_summary.txt ─────────────────────────

class _Tee:
    """Write every print() to both the terminal and a file."""
    def __init__(self, file_path):
        self._terminal = sys.stdout
        self._file     = open(file_path, 'w', encoding='utf-8')
    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)
    def flush(self):
        self._terminal.flush()
        self._file.flush()
    def close(self):
        self._file.close()

_tee = _Tee(os.path.join(OUT_DIR, 'sweep_summary.txt'))
sys.stdout = _tee

MUON_LOG  = os.path.join(V6_DIR, 'log_muon.txt')    # 6200-step baseline (fallback)
ADAMW_LOG = os.path.join(V6_DIR, 'log_adamw.txt')   # 6200-step baseline (fallback)

# ── Experiment definitions (must match sweep_config.py) ───────────────────────

sys.path.insert(0, V6_DIR)
from sweep_config import EXPERIMENTS, HYBRID_EXPERIMENTS

REG_COLORS = {
    'none':           'black',
    'sv_variance':    '#1f77b4',
    'orthogonal':     '#ff7f0e',
    'effective_rank': '#2ca02c',
    'stable_rank':    '#d62728',
    'isometry':       '#9467bd',
    'dead_sv':        '#8c564b',
    'hybrid':         '#17becf',    # teal — layer-specific Muon substitution
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
    num_iterations = None
    if not os.path.exists(log_path):
        return steps, losses, False, True   # missing
    with open(log_path) as f:
        for line in f:
            m = re.search(r'step:(\d+)/(\d+) val_loss:([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                num_iterations = int(m.group(2))
                losses.append(float(m.group(3)))
            if 'EARLY_STOP' in line:
                early_stopped = True
    # If the last logged step equals num_iterations the run completed fully —
    # ignore any EARLY_STOP lines (log contamination from old test runs)
    if steps and num_iterations and steps[-1] >= num_iterations:
        early_stopped = False
    return steps, losses, early_stopped, False


# ── Load baselines ────────────────────────────────────────────────────────────

# Auto-select Muon baseline will happen after SWEEP_END is known (see below)

# Auto-detect SWEEP_END from the experiments in sweep_config.py.
# Uses the most common num_iterations across ALL active experiments
# (including reg_name='none' runs like gflat, weight_proj, etc.)
# Excludes pure AdamW/Muon baselines (exp_name starts with 'adamw_' or 'muon_').
# Also folds in HYBRID_EXPERIMENTS so that running hybrids alone still gives the
# correct SWEEP_END even when all regular EXPERIMENTS are commented out.
_all_iters = [e['num_iterations'] for e in EXPERIMENTS
              if not e['exp_name'].startswith(('adamw_', 'muon_'))]
_all_iters += [e['num_iterations'] for e in HYBRID_EXPERIMENTS]
SWEEP_END  = max(set(_all_iters), key=_all_iters.count) if _all_iters else 6200
print(f'[INFO] Auto-detected SWEEP_END = {SWEEP_END}')

# Auto-select AdamW baseline with matching schedule:
#   ≤ 3000 iters → prefer log_adamw_3000.txt or logs/adamw_3000/log.txt
#   > 3000 iters → prefer log_adamw_{SWEEP_END}.txt, fall back to log_adamw.txt
_adamw_candidates = [
    (os.path.join(V6_DIR,   f'log_adamw_{SWEEP_END}.txt'),      f'log_adamw_{SWEEP_END}.txt'),
    (os.path.join(LOG_ROOT, f'adamw_{SWEEP_END}', 'log.txt'),   f'logs/adamw_{SWEEP_END}/log.txt'),
    (ADAMW_LOG,                                                   'log_adamw.txt (6200-step fallback)'),
]
adamw_log_path, adamw_log_name = next(
    ((p, n) for p, n in _adamw_candidates if os.path.exists(p)), (ADAMW_LOG, 'log_adamw.txt')
)
print(f'[INFO] AdamW baseline : {adamw_log_name}')
adamw_data = parse_baseline_log(adamw_log_path)

# Auto-select Muon baseline with matching schedule (same priority as AdamW)
_muon_candidates = [
    (os.path.join(V6_DIR,   f'log_muon_{SWEEP_END}.txt'),      f'log_muon_{SWEEP_END}.txt'),
    (os.path.join(LOG_ROOT, f'muon_{SWEEP_END}', 'log.txt'),   f'logs/muon_{SWEEP_END}/log.txt'),
    (MUON_LOG,                                                   'log_muon.txt (6200-step fallback)'),
]
muon_log_path, muon_log_name = next(
    ((p, n) for p, n in _muon_candidates if os.path.exists(p)), (MUON_LOG, 'log_muon.txt')
)
print(f'[INFO] Muon  baseline : {muon_log_name}')
muon_data = parse_baseline_log(muon_log_path)

# Load corrected AdamW baseline (3000 steps, warmdown=871 — correct proportion)
_adamw_correct_log = os.path.join(LOG_ROOT, 'adamw_3000_correct_warmdown', 'log.txt')
_ac_steps, _ac_losses, _, _ac_missing = parse_exp_log(_adamw_correct_log)
adamw_correct_data  = dict(zip(_ac_steps, _ac_losses))
adamw_correct_final = _ac_losses[-1] if _ac_losses else float('nan')
_adamw_correct_avail = not _ac_missing and bool(_ac_losses)
if _adamw_correct_avail:
    print(f'[INFO] AdamW corrected : logs/adamw_3000_correct_warmdown/log.txt')
else:
    print(f'[INFO] AdamW corrected : NOT YET AVAILABLE (run adamw_3000_correct_warmdown)')

# ── Filter experiments by phase ───────────────────────────────────────────────
# Phase 1 (SWEEP_END ≤ 3000): exclude full_* runs (they use a longer schedule)
# Phase 2 (SWEEP_END > 3000): exclude non-full_* runs (they are short sweeps)
if SWEEP_END <= 3000:
    _before = len(EXPERIMENTS)
    EXPERIMENTS = [e for e in EXPERIMENTS
                   if not e['exp_name'].startswith('full_')
                   and e['exp_name'] not in (f'adamw_{SWEEP_END}', 'adamw_3000_correct_warmdown')]
    _skip = _before - len(EXPERIMENTS)
    if _skip:
        print(f'[INFO] Phase 1 mode: skipped {_skip} full_* experiment(s)')
else:
    _before = len(EXPERIMENTS)
    EXPERIMENTS = [e for e in EXPERIMENTS
                   if e['exp_name'].startswith('full_') or e['reg_name'] == 'none']
    _skip = _before - len(EXPERIMENTS)
    if _skip:
        print(f'[INFO] Phase 2 mode: skipped {_skip} non-full_* experiment(s)')

muon_steps   = sorted(muon_data.keys())
adamw_steps  = sorted(adamw_data.keys())
muon_losses  = [muon_data[s]  for s in muon_steps]
adamw_losses = [adamw_data[s] for s in adamw_steps]

muon_final  = muon_data.get(SWEEP_END,  muon_losses[-1])
adamw_final = adamw_data.get(SWEEP_END, adamw_losses[-1])
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


# ── Load hybrid experiment results ───────────────────────────────────────────

for hexp in HYBRID_EXPERIMENTS:
    name     = hexp['exp_name']
    log_path = os.path.join(LOG_ROOT, name, 'log.txt')
    steps, losses, early_stopped, missing = parse_exp_log(log_path)

    fl    = losses[-1] if losses else float('nan')
    ls    = steps[-1]  if steps  else 0
    d_adamw = fl - adamw_final if not np.isnan(fl) else float('nan')
    d_muon  = fl - muon_final  if not np.isnan(fl) else float('nan')
    gap_closed = (d_adamw / muon_gap * -100) if (not np.isnan(d_adamw) and muon_gap > 0) else float('nan')

    # 'layer' group = all matrix types, vary which layers get Muon
    # 'matrix' group = all layers, vary which matrix type gets Muon
    _group = 'layer' if hexp.get('muon_matrices') == 'all' else 'matrix'

    results.append({
        'exp_name':      name,
        'reg_name':      'hybrid',
        'lam':           0.0,
        'steps':         steps,
        'losses':        losses,
        'final_loss':    fl,
        'last_step':     ls,
        'd_adamw':       d_adamw,
        'd_muon':        d_muon,
        'gap_closed':    gap_closed,
        'early_stopped': early_stopped,
        'missing':       missing,
        '_description':  hexp.get('description', name),
        '_group':        _group,
    })


# ── Console table ─────────────────────────────────────────────────────────────

COL    = 34   # experiment name column width
REG_W  = 16   # regularizer column width
DESC_W = 30   # λ / description column width (wide enough for longest hybrid description)
TBL_W  = COL + 1 + REG_W + 1 + DESC_W + 2 + 5 + 2 + 7 + 2 + 7 + 2 + 7 + 2 + 6  # = 126

_correct_gap = adamw_correct_final - muon_final if _adamw_correct_avail else float('nan')
_correct_str = (f'   AdamW✓ = {adamw_correct_final:.4f}   Gap✓ = {_correct_gap:.4f}'
                if _adamw_correct_avail else '   AdamW✓ = (pending)')
print()
print('=' * TBL_W)
print(f'  Baselines at step {SWEEP_END}:   '
      f'AdamW = {adamw_final:.4f}   '
      f'Muon = {muon_final:.4f}   '
      f'Gap = {muon_gap:.4f}'
      f'{_correct_str}')
print('=' * TBL_W)
print(f'{"Experiment":<{COL}} {"Reg":<{REG_W}} {"λ / Description":<{DESC_W}}  {"Steps":>5}  '
      f'{"ValLoss":>7}  {"ΔAdamW":>7}  {"ΔMuon":>7}  {"Gap%":>6}  Status')
print('-' * TBL_W)

for r in sorted(results, key=lambda x: x['final_loss']):
    status = ('MISSING'    if r['missing']
              else 'STOPPED' if r['early_stopped']
              else 'OK')
    fl_s  = f'{r["final_loss"]:.4f}' if not np.isnan(r['final_loss']) else '—'
    da_s  = f'{r["d_adamw"]:+.4f}'   if not np.isnan(r['d_adamw'])   else '—'
    dm_s  = f'{r["d_muon"]:+.4f}'    if not np.isnan(r['d_muon'])    else '—'
    gc_s  = f'{r["gap_closed"]:+.1f}%' if not np.isnan(r['gap_closed']) else '—'
    if r['reg_name'] == 'hybrid':
        desc = r.get('_description', r['exp_name'])[:DESC_W]
        lam_col = f'{desc:<{DESC_W}}'
    else:
        lam_col = f'{r["lam"]:>{DESC_W}.0e}'
    print(f'{r["exp_name"]:<{COL}} {r["reg_name"]:<{REG_W}} {lam_col}  '
          f'{r["last_step"]:>5}  {fl_s:>7}  {da_s:>7}  {dm_s:>7}  {gc_s:>6}  {status}')

print('-' * TBL_W)
_dash = f'{"—":<{DESC_W}}'
print(f'{"AdamW baseline (wrong warmdown)":<{COL}} {"—":<{REG_W}} {_dash}  {SWEEP_END:>5}  '
      f'{adamw_final:>7.4f}  {"0.0000":>7}  {(-muon_gap):>7.4f}  {"0.0%":>6}  REF  ← {adamw_log_name}')
if _adamw_correct_avail:
    _d_correct  = adamw_correct_final - adamw_final
    _gc_correct = (_d_correct / muon_gap * -100) if muon_gap > 0 else float('nan')
    print(f'{"AdamW baseline (correct warmdown)":<{COL}} {"—":<{REG_W}} {_dash}  {SWEEP_END:>5}  '
          f'{adamw_correct_final:>7.4f}  {_d_correct:>+7.4f}  {(adamw_correct_final-muon_final):>+7.4f}  '
          f'{_gc_correct:>+5.1f}%  REF✓')
else:
    print(f'{"AdamW baseline (correct warmdown)":<{COL}} {"—":<{REG_W}} {_dash}  {"—":>5}  '
          f'{"(pending)":>7}  {"—":>7}  {"—":>7}  {"—":>6}  REF✓ (not yet run)')
print(f'{"Muon baseline (target)":<{COL}} {"—":<{REG_W}} {_dash}  {SWEEP_END:>5}  '
      f'{muon_final:>7.4f}  {(-muon_gap):>7.4f}  {"0.0000":>7}  {"100.0%":>6}  TARGET  ← {muon_log_name}')
print('=' * TBL_W)


# ── Detailed trajectory (every 100 steps) ────────────────────────────────────

# Lookups: {step: loss}
_traj     = {r['exp_name']: dict(zip(r['steps'], r['losses'])) for r in results}
_adamw_lut = dict(zip(adamw_steps, adamw_losses))
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
print('─' * TBL_W)
print('  Val-loss trajectory  (every 100 steps)   ↓ = below AdamW at that step')
print('─' * TBL_W)

# ── Print AdamW, AdamW✓ & Muon reference rows once ──────────────────────────
_adamw_correct_lut = dict(zip(_ac_steps, _ac_losses))
_ref_series = [('  AdamW ref (wrong wd)', _adamw_lut), ('   Muon ref', _muon_lut)]
if _adamw_correct_avail:
    _ref_series.insert(1, ('AdamW✓ ref (correct wd)', _adamw_correct_lut))
for ref_label, ref_lut in _ref_series:
    ref_tokens = []
    for s in _all_steps:
        v = ref_lut.get(s)
        ref_tokens.append(f'{s:>4}:{v:.4f} ' if v is not None else f'{s:>4}:  —    ')
    if ref_tokens:
        print(f'\n{ref_label}  ·  baseline')
        for i in range(0, len(ref_tokens), _PER_ROW):
            print('    ' + '  '.join(ref_tokens[i:i + _PER_ROW]))

print()
print('─' * TBL_W)

# ── Print one block per experiment, grouped by regularizer ───────────────────
cur_reg = None
for r in results:          # keep original EXPERIMENTS order
    lut = _traj[r['exp_name']]

    # Separator between regularizer families
    if r['reg_name'] != cur_reg:
        cur_reg = r['reg_name']
        print(f'\n  ── {cur_reg} ──')

    status_tag = '  [STOPPED]' if r['early_stopped'] else ('  [MISSING]' if r['missing'] else '')
    if r['reg_name'] == 'hybrid':
        desc = r.get('_description', r['exp_name'])
        print(f'\n  {r["exp_name"]}  ·  {desc}{status_tag}')
    else:
        print(f'\n  {r["exp_name"]}  ·  λ={r["lam"]:.0e}{status_tag}')

    tokens = _traj_rows(lut, r['exp_name'], _adamw_lut, _all_steps)
    if not tokens:
        print('    (no data)')
        continue
    for i in range(0, len(tokens), _PER_ROW):
        print('    ' + '  '.join(tokens[i:i + _PER_ROW]))

print()
print('─' * TBL_W)


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

    # Baselines — always show old AdamW (reg exps were built on it) + correct if available
    ax.plot(adamw_steps, adamw_losses, color='#C44E52', linewidth=2.2,
            linestyle='--', label=f'AdamW wrong wd ({adamw_final:.4f})', zorder=6)
    ax.plot(muon_steps,  muon_losses,  color='#4C72B0', linewidth=2.2,
            linestyle='--', label=f'Muon baseline ({muon_final:.4f})', zorder=6)
    if _adamw_correct_avail:
        ax.plot(_ac_steps, _ac_losses, color='#e67e22', linewidth=2.2,
                linestyle='--', label=f'AdamW✓ ({adamw_correct_final:.4f})', zorder=6)

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
labels = [r.get('_description', r['exp_name']) if r['reg_name'] == 'hybrid'
          else f'{r["reg_name"]}  λ={r["lam"]:.0e}'
          for r in valid]
values = [r['final_loss'] for r in valid]
colors = [REG_COLORS.get(r['reg_name'], 'gray') for r in valid]

bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(adamw_final, color='#C44E52', linewidth=2.0, linestyle='--',
           label=f'AdamW wrong wd  {adamw_final:.4f}')
ax.axvline(muon_final,  color='#4C72B0', linewidth=2.0, linestyle='--',
           label=f'Muon baseline   {muon_final:.4f}')
if _adamw_correct_avail:
    ax.axvline(adamw_correct_final, color='#e67e22', linewidth=2.0, linestyle='--',
               label=f'AdamW✓  {adamw_correct_final:.4f}')
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


# ── Plot 3: Hybrid Muon+AdamW — val-loss curves ───────────────────────────────

hybrid_results = [r for r in results if r['reg_name'] == 'hybrid']

if hybrid_results:
    layer_group  = [r for r in hybrid_results if r.get('_group') == 'layer']
    matrix_group = [r for r in hybrid_results if r.get('_group') == 'matrix']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharey=True)

    # Use tab10 for both panels — 10 perceptually distinct colours
    _tab10 = cm.get_cmap('tab10')
    _DISTINCT = [_tab10(i) for i in range(10)]

    _groups = [
        (axes[0], layer_group,  'Layer Depth Sweep\n(all matrix types → Muon, vary layers)'),
        (axes[1], matrix_group, 'Matrix Type Sweep\n(all layers, vary which matrices → Muon)'),
    ]

    for ax, group, title in _groups:
        # AdamW / Muon baselines
        ax.plot(adamw_steps, adamw_losses, color='#C44E52', linewidth=2.2,
                linestyle='--', zorder=6, label=f'AdamW wrong wd  {adamw_final:.4f}')
        ax.plot(muon_steps,  muon_losses,  color='#4C72B0', linewidth=2.2,
                linestyle='--', zorder=6, label=f'Muon   {muon_final:.4f}')
        if _adamw_correct_avail:
            ax.plot(_ac_steps, _ac_losses, color='#e67e22', linewidth=2.2,
                    linestyle='--', zorder=6, label=f'AdamW✓  {adamw_correct_final:.4f}')
        ax.axhspan(muon_final, adamw_final, color='grey', alpha=0.08)

        for i, run in enumerate(group):
            if not run['steps']:
                continue
            col  = _DISTINCT[i % len(_DISTINCT)]
            desc = run.get('_description', run['exp_name'])
            fl_s = f'{run["final_loss"]:.4f}' if not np.isnan(run['final_loss']) else '—'
            gc_s = (f'  {run["gap_closed"]:+.1f}%' if not np.isnan(run['gap_closed']) else '')
            style = ':' if run['early_stopped'] else '-'
            ax.plot(run['steps'], run['losses'], color=col, linewidth=1.9,
                    linestyle=style, label=f'{desc}  →  {fl_s}{gc_s}')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Val Loss', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, SWEEP_END + 100)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f'Hybrid Muon+AdamW — Layer & Matrix Sweep  (step {SWEEP_END})\n'
        f'AdamW = {adamw_final:.4f}   Muon = {muon_final:.4f}   Gap = {muon_gap:.4f}',
        fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'hybrid_curves.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')


# ── Plot 4: Hybrid — gap-closed bar chart ─────────────────────────────────────

    valid_hybrid = [r for r in hybrid_results
                    if not r['missing'] and not np.isnan(r['gap_closed'])]

    if valid_hybrid:
        # Order: layer group first (sorted by last_step desc as tiebreak → by exp order),
        # then matrix group — keep the natural order from HYBRID_EXPERIMENTS
        ordered = [r for r in hybrid_results if r in valid_hybrid]

        fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.55 + 2.0)), dpi=150)

        bar_labels = [r.get('_description', r['exp_name']) for r in ordered]
        bar_values = [r['gap_closed'] for r in ordered]
        bar_colors = ['#4C9BE8' if r.get('_group') == 'layer' else '#3CB371'
                      for r in ordered]

        bars = ax.barh(bar_labels, bar_values, color=bar_colors,
                       edgecolor='white', linewidth=0.5)
        ax.bar_label(bars, fmt='%+.1f%%', padding=4, fontsize=9)

        ax.axvline(0,   color='#C44E52', linewidth=2.0, linestyle='--',
                   label=f'AdamW baseline (0%)')
        ax.axvline(100, color='#4C72B0', linewidth=2.0, linestyle='--',
                   label=f'Muon target (100%)')
        ax.axvspan(0, 100, color='grey', alpha=0.06)

        # Custom legend for groups
        from matplotlib.patches import Patch
        ax.legend(handles=[
            ax.get_lines()[0], ax.get_lines()[1],
            Patch(color='#4C9BE8', label='Layer sweep (all matrices)'),
            Patch(color='#3CB371', label='Matrix sweep (all layers)'),
        ], fontsize=9, loc='lower right')

        ax.set_xlabel('Gap closed toward Muon  (%)', fontsize=10)
        ax.set_title(
            f'Hybrid Muon+AdamW — Fraction of Muon Gap Recovered  (step {SWEEP_END})\n'
            f'0% = AdamW level  |  100% = full Muon  |  Gap = {muon_gap:.4f}',
            fontsize=11,
        )
        ax.grid(True, axis='x', alpha=0.25)
        lo = min(bar_values + [0]) - 10
        hi = max(bar_values + [100]) + 15
        ax.set_xlim(lo, hi)
        ax.invert_yaxis()
        fig.tight_layout()
        out = os.path.join(OUT_DIR, 'hybrid_gap.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')

# ── Close tee ─────────────────────────────────────────────────────────────────
sys.stdout = _tee._terminal
_tee.close()
print(f'Console output saved to: {os.path.join(OUT_DIR, "sweep_summary.txt")}')
