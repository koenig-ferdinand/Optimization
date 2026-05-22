# PACKAGES
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# FILES
import functions

# MAIN.PY
def load_loss(path, loss_type):
    steps = []
    losses = []
    with open(path) as f:
        for line in f:
            if loss_type not in line:
                continue
            parts = line.split()
            step = int(parts[0].split(":")[1].split("/")[0])
            loss = float(parts[1].split(":")[1])
            steps.append(step)
            losses.append(loss)
    return steps, losses

# load all three
adam_base_steps, adam_base_loss = load_loss('data/a_adam_baseline/log.txt', 'val_loss')
adam_new_steps,  adam_new_loss  = load_loss('data/a_adam_new/log.txt',      'val_loss')
muon_base_steps, muon_base_loss = load_loss('data/a_muon_baseline/log.txt', 'val_loss')

# truncate all runs to the shortest one
cutoff = min(adam_base_steps[-1], adam_new_steps[-1], muon_base_steps[-1])

def clip(steps, losses, cutoff):
    steps  = np.array(steps)
    losses = np.array(losses)
    mask   = steps <= cutoff
    return steps[mask], losses[mask]

adam_base_steps, adam_base_loss = clip(adam_base_steps, adam_base_loss, cutoff)
adam_new_steps,  adam_new_loss  = clip(adam_new_steps,  adam_new_loss,  cutoff)
muon_base_steps, muon_base_loss = clip(muon_base_steps, muon_base_loss, cutoff)

# sanity check on the truncated arrays
assert np.array_equal(adam_base_steps, adam_new_steps) \
   and np.array_equal(adam_base_steps, muon_base_steps), "Step grids differ within range!"

steps = adam_base_steps
diff  = adam_base_loss - adam_new_loss

# plotting: two side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), dpi=600)

# --- left: all three val losses ---
ax1.plot(steps, adam_base_loss, linewidth=0.8, alpha=0.8, label='AdamW baseline')
ax1.plot(steps, adam_new_loss,  linewidth=0.8, alpha=0.8, label='AdamW new')
ax1.plot(steps, muon_base_loss, linewidth=0.8, alpha=0.8, label='Muon baseline')
ax1.set_xlabel("Step")
ax1.set_ylabel("Validation loss")
ax1.set_title("Validation loss")
ax1.legend()
ax1.xaxis.set_major_locator(MultipleLocator(500))
ax1.xaxis.set_minor_locator(MultipleLocator(100))
ax1.grid(True, which="major", alpha=0.4)
ax1.grid(True, which="minor", alpha=0.15)

# --- right: difference baseline - new (AdamW) ---
ax2.plot(steps, diff, linewidth=0.8, alpha=0.8)
ax2.set_xlabel("Step")
ax2.set_title("Validation loss difference: AdamW baseline - AdamW new")
ax2.xaxis.set_major_locator(MultipleLocator(500))
ax2.xaxis.set_minor_locator(MultipleLocator(100))
ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))
ax2.yaxis.set_minor_locator(plt.MaxNLocator(nbins=40))
ax2.grid(True, which="major", alpha=0.4)
ax2.grid(True, which="minor", alpha=0.15)

fig.tight_layout()
fig.savefig("analyse/plots/3000base_iso_e-5_val_loss_comparison.png")