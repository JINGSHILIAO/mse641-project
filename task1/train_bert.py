import torch
from torch.types import Device
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from tqdm.auto import tqdm
from dataset import get_dataloader


# Configurations
# MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = "roberta-base"
TRAIN_PATH = 'data/train.jsonl'
VAL_PATH = 'data/val.jsonl'
BATCH_SIZE = 4
MAX_LEN = 64
# MAX_LEN = 256 # when including paraghraph context
EPOCHS = 3
LR = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loaders
train_loader = get_dataloader(TRAIN_PATH, tokenizer_name=MODEL_NAME, batch_size=BATCH_SIZE, max_len=MAX_LEN)
val_loader = get_dataloader(VAL_PATH, tokenizer_name=MODEL_NAME, batch_size=BATCH_SIZE, max_len=MAX_LEN, is_test=False)

# Compute and set class weights
train_counts = {'phrase': 1367, 'passage': 1274, 'multi': 559}
total_samples = sum(train_counts.values())
class_weights = {label: total_samples/count for label, count in train_counts.items()}
sum_weights = sum(class_weights.values())
num_classes = len(class_weights)
normalized_weights = [
    (class_weights['phrase'] / sum_weights) * num_classes,
    (class_weights['passage'] / sum_weights) * num_classes,
    (class_weights['multi'] / sum_weights) * num_classes
]
class_weights_tensor = torch.tensor(normalized_weights).to(DEVICE)
print("Using Class Weights:", class_weights_tensor)

# Model initilization
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(DEVICE)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(0.1*total_steps), 
    num_training_steps=total_steps
)

# Loss function with class weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# Training Function
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, f1

# Evaluation Function
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Detailed classification report for diagnostics
    print(classification_report(all_labels, all_preds, target_names=['phrase', 'passage', 'multi']))

    return avg_loss, acc, f1

# Main Training Loop
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
    print(f"[Train] Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | Macro-F1: {train_f1:.4f}")

    val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, DEVICE)
    print(f"[Val] Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")

# Save the model after training
model.save_pretrained('roberta_base_with_context_3_epochs')
