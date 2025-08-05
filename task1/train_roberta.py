import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import ClickbaitSpoilerTypeDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import torch.nn as nn

# Run from terminal
parser = argparse.ArgumentParser()
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--output_dir", type=str, default="runs/roberta_default")
args = parser.parse_args()

MODEL_NAME = "roberta-base"
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/val.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_dataset = ClickbaitSpoilerTypeDataset(TRAIN_PATH, tokenizer_name=MODEL_NAME, max_len=MAX_LEN)
val_dataset = ClickbaitSpoilerTypeDataset(VAL_PATH, tokenizer_name=MODEL_NAME, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.config.hidden_dropout_prob = args.dropout
model.to(DEVICE)

class_weights = torch.tensor([0.6639, 0.7124, 1.6236], device=DEVICE)  # from earlier
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=args.weight_decay)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1

best_f1 = 0
patience = 2
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="macro")

    val_loss, val_acc, val_f1 = evaluate(model, val_loader)

    print(f"[Train] Loss: {running_loss/len(train_loader):.4f} | Accuracy: {train_acc:.4f} | Macro-F1: {train_f1:.4f}")
    print(f"[Val]   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({
                "val_acc": val_acc,
                "val_f1": val_f1,
                "train_acc": train_acc,
                "train_f1": train_f1
            }, f)
        print("Saved best model so far")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
