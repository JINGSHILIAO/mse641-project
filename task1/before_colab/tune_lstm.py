import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from itertools import product
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.rnn_classifier_v2 import RNNClassifier
from dataset_v2 import get_dataloader

# Fixed settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
PATIENCE = 3
VOCAB_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"
MAX_LEN = 200
BATCH_SIZE = 32

# Tuning grid
hidden_dims = [64, 128, 256]
dropouts = [0.3, 0.5, 0.6]
learning_rates = [1e-3, 5e-4, 1e-4]
all_configs = list(product(hidden_dims, dropouts, learning_rates))

results = []
best_val_acc = 0.0
best_model_state = None
best_config = None

train_loader, vocab = get_dataloader(VOCAB_PATH, batch_size=BATCH_SIZE, max_len=MAX_LEN)
val_loader, _ = get_dataloader(VAL_PATH, vocab=vocab, batch_size=BATCH_SIZE, max_len=MAX_LEN)

output_dir = "outputs_lstm"
os.makedirs(output_dir, exist_ok=True)

# Start grid search
for i, (hidden_dim, dropout, lr) in enumerate(all_configs, 1):
    config = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": lr,
        "embed_dim": 100,
        "l2_reg": 1e-5,
        "batch_size": BATCH_SIZE
    }

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_layers=1,
        bidirectional=False
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["l2_reg"])
    criterion = nn.CrossEntropyLoss()

    print(f"[{i}/{len(all_configs)}] Training LSTM | hidden_dim={hidden_dim} | dropout={dropout} | lr={lr}")
    best_run_val_acc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        elapsed = round(time.time() - start_time, 2)

        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        results.append({
            "config_id": i,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": lr,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_time_sec": elapsed
        })

        # Early stopping check
        if val_acc > best_run_val_acc:
            best_run_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        # Save best model overall
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_config = config.copy()

# Save best model
if best_model_state:
    torch.save(best_model_state, f"{output_dir}/best_lstm_model.pt")
    print(f"Best model saved with val_acc={best_val_acc:.4f} | Config: {best_config}")

# Save results
df = pd.DataFrame(results)
df.to_csv(f"{output_dir}/tune_lstm_results.csv", index=False)
print(f"Grid search results saved to {output_dir}/tune_lstm_results.csv")
