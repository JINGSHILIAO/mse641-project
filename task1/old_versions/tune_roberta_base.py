
import os
import time
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from models.transformer_classifier import TransformerClassifier
from transformer_based_dataset_big import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("task1/outputs", exist_ok=True)

# Grid search configurations
param_grid = [
    {"max_len": 128, "batch_size": 32, "learning_rate": 2e-5, "dropout": 0.1},
    {"max_len": 256, "batch_size": 16, "learning_rate": 2e-5, "dropout": 0.3},
    {"max_len": 256, "batch_size": 16, "learning_rate": 3e-5, "dropout": 0.1},
    {"max_len": 128, "batch_size": 32, "learning_rate": 1e-5, "dropout": 0.3},
    {"max_len": 256, "batch_size": 32, "learning_rate": 1e-5, "dropout": 0.2},
]

best_f1 = 0
best_config = None
summary_logs = []

for config in param_grid:
    print(f"Testing config: {config}")
    train_loader = get_dataloader("data/train.jsonl", batch_size=config["batch_size"],
                                  tokenizer_name="roberta-base", max_len=config["max_len"])
    val_loader = get_dataloader("data/val.jsonl", batch_size=config["batch_size"],
                                tokenizer_name="roberta-base", max_len=config["max_len"])

    model = TransformerClassifier(model_name="roberta-base", num_labels=3, dropout=config["dropout"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=5 * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0
    early_stop_counter = 0
    log_rows = []
    start_run = time.time()

    for epoch in range(1, 5):  # 4 epochs max
        model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = (torch.tensor(val_preds) == torch.tensor(val_labels)).float().mean().item()
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_loss /= len(val_loader)
        duration = time.time() - start_time

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {duration:.2f}s")
        log_rows.append([epoch, train_loss, val_loss, val_acc, val_f1, duration])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "task1/outputs/best_roberta_tuned_model.pt")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 2:
                print("Early stopping triggered.")
                break

    duration = time.time() - start_run
    summary_logs.append({
        "max_len": config["max_len"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "dropout": config["dropout"],
        "best_val_f1": best_val_f1,
        "time": duration
    })

    if best_val_f1 > best_f1:
        best_f1 = best_val_f1
        best_config = config

# Save all config results
summary_df = pd.DataFrame(summary_logs)
summary_df.to_csv("task1/outputs/roberta_gridsearch_summary.csv", index=False)

# Print best config
print(f"Best Val F1: {best_f1:.4f} with config: {best_config}")
