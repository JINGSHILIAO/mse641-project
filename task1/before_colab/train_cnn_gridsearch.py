
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import csv
import os
import sys
from itertools import product

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
MAX_LEN = 30
PATIENCE = 5

LRs = [0.001, 0.0005]
DROPOUTs = [0.3, 0.5]
L2_REGs = [1e-5, 1e-4, 1e-3]


train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=BATCH_SIZE)
val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=BATCH_SIZE)


csv_path = "grid_search_results.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["LR", "Dropout", "L2_Regularization", "BestEpoch", "BestTrainAcc", "BestValAcc", "EpochsRun"])

    for lr, dropout, l2 in product(LRs, DROPOUTs, L2_REGs):
        logger.info(f"Training config: LR={lr}, Dropout={dropout}, L2={l2}")
        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            dropout=dropout,
            padding_idx=vocab["<PAD>"]
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        best_val_acc = 0.0
        best_train_acc = 0.0
        best_epoch = 0
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
                best_train_acc = train_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        writer.writerow([lr, dropout, l2, best_epoch, round(best_train_acc, 4), round(best_val_acc, 4), epoch + 1])
        logger.info("Finished config.\n")
