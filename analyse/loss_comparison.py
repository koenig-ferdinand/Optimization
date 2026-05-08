# PACKAGES
import torch
import matplotlib.pyplot as plt
import numpy as np 
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

adam_train_steps, adam_train_loss =  load_loss('data/adamw_prewarm/losses.txt', 'train_loss')
adam_val_steps, adam_val_loss = load_loss('data/adamw_prewarm/losses.txt', 'val_loss')

muon_train_steps, muon_train_loss =  load_loss('data/muon_updated/losses.txt', 'train_loss')
muon_val_steps, muon_val_loss =  load_loss('data/muon_updated/losses.txt', 'val_loss')

# plotting
fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 4.5), dpi = 600)

# train_loss
ax_train.plot(adam_train_steps, adam_train_loss, label="AdamW", linewidth = 0.8, alpha=0.8)
ax_train.plot(muon_train_steps, muon_train_loss, label="Muon", linewidth = 0.8, alpha=0.8)
ax_train.set_yscale("log")
ax_train.set_xlabel("Step")
ax_train.set_ylabel("Training loss")
ax_train.set_title("Training loss")
ax_train.grid(True, which="both", alpha=0.3)
ax_train.legend()

# val_loss
ax_val.plot(adam_val_steps, adam_val_loss, label="AdamW", marker="o", markersize=3)
ax_val.plot(muon_val_steps, muon_val_loss, label="Muon", marker="x", markersize=3)
ax_val.set_yscale("log")
ax_val.set_xlabel("Step")
ax_val.set_ylabel("Validation loss")
ax_val.set_title("Validation loss")
ax_val.grid(True, which="both", alpha=0.3)
ax_val.legend()

fig.suptitle("Losses of AdamW and Muon")
fig.tight_layout()
fig.savefig("analyse/plots/loss_comparison.png")

