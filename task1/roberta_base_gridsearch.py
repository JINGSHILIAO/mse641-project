import os
import itertools

# --- Grid values to search ---
dropouts = [0.1, 0.2, 0.3]
weight_decays = [0.0, 0.01, 0.05]

# --- Other config ---
epochs = 6  # Use early stopping inside training script
base_cmd = "python task1/train_roberta.py"

# --- Run all combinations ---
for dropout, wd in itertools.product(dropouts, weight_decays):
    out_dir = f"runs/roberta_d{dropout}_wd{wd}".replace('.', '')
    cmd = (
        f"{base_cmd} "
        f"--dropout {dropout} "
        f"--weight_decay {wd} "
        f"--epochs {epochs} "
        f"--output_dir {out_dir}"
    )
    print(f"Running: {cmd}")
    os.system(cmd)