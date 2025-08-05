from pickle import FALSE
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from time import time
import pandas as pd

from bert_dataset import get_bert_dataloader
from models.transformer_classifier import TransformerClassifier

# model_name = "distilbert-base-uncased"
model_name = "bert-base-uncased"
train_path = "data/train.jsonl"
val_path = "data/val.jsonl"
batch_size = 32
max_len = 64
lr = 2e-5
epochs = 5
early_stopping_patience = 2
# csv_log_path = "outputs/train_distilbert_log.csv"
csv_log_path = "outputs/train_bert_base_uncased_log.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_bert_dataloader(train_path, batch_size, model_name, max_len)
val_loader = get_bert_dataloader(val_path, batch_size, model_name, max_len, is_test=False)

model = TransformerClassifier(model_name=model_name, num_labels=3)
model.to(device)

optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

os.makedirs("outputs", exist_ok=True)
log_rows = []
best_val_f1 = 0
epochs_no_improve = 0

print("Training on:", device)
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    start_time = time()

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    epoch_time = round(time() - start_time, 2)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | ⏱️ Time: {epoch_time}s")

    log_rows.append([epoch, train_loss, val_loss, val_acc, val_f1, epoch_time])
    pd.DataFrame(log_rows, columns=["epoch", "train_loss", "val_loss", "val_accuracy", "val_f1", "train_time_sec"]).to_csv(csv_log_path, index=False)

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), "outputs/best_bert_base_uncased_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Training complete. Best Val F1:", round(best_val_f1, 4))
