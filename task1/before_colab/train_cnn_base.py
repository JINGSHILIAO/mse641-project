import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.cnn_classifier_task1 import CNNClassifier
from scripts.task1.dataset import get_dataloader, label_map, reverse_label_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

BATCH_SIZE = 32
EMBED_DIM = 100
NUM_CLASSES = 3
NUM_EPOCHS = 50
LR = 0.005
DROPOUT = 0.3
MAX_LEN = 30
L2_REG = 0.001  
PATIENCE = 5        # Early stopping if no improvement for 5 epochs


train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=BATCH_SIZE)
val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=BATCH_SIZE)

model = CNNClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
    padding_idx=vocab["<PAD>"]
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG)

best_val_acc = 0.0
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += inputs.size(0)

    train_acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

    
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += inputs.size(0)

    val_acc = val_correct / val_total
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        logger.info("Validation accuracy improved. Resetting early stopping counter.")
        # torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_no_improve += 1
        logger.info(f"No improvement. Patience counter: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            logger.info("Early stopping triggered.")
            break