
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import f1_score
import pandas as pd
from models.transformer_classifier import TransformerClassifier
from transformer_based_dataset import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
model_name = "microsoft/deberta-v3-small"
max_len = 64
batch_size = 16 # reduce batch size to avoid OOM due to more complexity
learning_rate = 2e-5
epochs = 5
patience = 2

# Load data
train_loader = get_dataloader("data/train.jsonl", batch_size=batch_size,
                                   tokenizer_name=model_name, max_len=max_len)
val_loader = get_dataloader("data/val.jsonl", batch_size=batch_size,
                                 tokenizer_name=model_name, max_len=max_len)

# Model
model = TransformerClassifier(model_name=model_name, num_labels=3).to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=epochs * len(train_loader))

# Loss
criterion = nn.CrossEntropyLoss()

# Logging
log_rows = []
best_val_f1 = 0
early_stop_counter = 0
os.makedirs("task1/outputs", exist_ok=True)

print(f"Training on device: {device}")

for epoch in range(1, epochs + 1):
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
        torch.save(model.state_dict(), "task1/outputs/best_deberta_v3_small_model.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

print(f"Training complete. Best Val F1: {best_val_f1:.4f}")
df = pd.DataFrame(log_rows, columns=["Epoch", "Train Loss", "Val Loss", "Val Acc", "Val F1", "Time"])
df.to_csv("task1/outputs/train_deberta_log.csv", index=False)
