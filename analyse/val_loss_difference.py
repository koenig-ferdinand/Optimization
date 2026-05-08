# PACKAGES
import torch
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.ticker import MultipleLocator

# -------------------------------------------------------------------------------------------------

# FILES
import functions
# -------------------------------------------------------------------------------------------------

# SETUP: 
# functions file contains modular functions (svd, stable_rank, effective_rank,...)
# configure new modular functions in main/ separate files
# -------------------------------------------------------------------------------------------------
 

# MAIN.PY
# loss type either train_loss or val_loss 
def load_loss(path, loss_type): 
    steps = []
    losses = []

    with open(path) as f: 
        for line in f: 
            if loss_type not in line: 
                continue
           
            # line looks like: "step:100/6200 val_loss:5.3992 train_time:108768ms ..."
            # Split it into pieces separated by spaces
            parts = line.split()

            step = int(parts[0].split(":")[1].split("/")[0])
            loss = float(parts[1].split(":")[1])

            steps.append(step)
            losses.append(loss)

    return steps, losses

adam_val_steps, adam_val_loss = load_loss('data/adamw_prewarm/losses.txt', 'val_loss')
muon_val_steps, muon_val_loss =  load_loss('data/muon_updated/losses.txt', 'val_loss')

assert muon_val_steps == adam_val_steps, "Validation steps don't match!"

steps = np.array(adam_val_steps)
diff = np.array(adam_val_loss) - np.array(muon_val_loss)

# plotting
fig, ax = plt.subplots(figsize=(8, 4.5), dpi = 600)
ax.plot(steps, diff, linewidth=0.8, alpha=0.8)
ax.set_ylim(bottom=0)

ax.set_xlabel("Step")
ax.set_title("Validation loss difference: AdamW - Muon")
ax.grid(True, alpha=0.3)

# --- ticks (the numbers shown on the axes) ---
ax.xaxis.set_major_locator(MultipleLocator(500))   # major x-ticks every 500 steps
ax.xaxis.set_minor_locator(MultipleLocator(100))   # minor x-ticks every 100 steps
ax.yaxis.set_major_locator(MultipleLocator(0.1))   # major y-ticks every 0.1
ax.yaxis.set_minor_locator(MultipleLocator(0.02))  # minor y-ticks every 0.02

# --- grid (lines in the background) ---
ax.grid(True, which="major", alpha=0.4)
ax.grid(True, which="minor", alpha=0.15)   # fainter for the finer grid

fig.tight_layout()
fig.savefig("analyse/plots/val_loss_diff.png")