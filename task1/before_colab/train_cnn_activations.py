import torch
import torch.nn as nn
import torch.optim as optim
import logging
import csv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.cnn_classifier_task1 import CNNClassifier
from scripts.task1.dataset import get_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

BATCH_SIZE = 32
EMBED_DIM = 100
NUM_CLASSES = 3
NUM_EPOCHS = 50
LR = 0.001
DROPOUT = 0.3
MAX_LEN = 30
L2_REG = 0.001
PATIENCE = 5

train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=BATCH_SIZE)
val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=BATCH_SIZE)

activations = ["relu", "gelu", "leaky_relu", "tanh", "sigmoid"]

overall_best_val_acc = 0.0
overall_best_model = None
overall_best_act = ""

csv_path = "activation_results.csv"
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Activation", "BestEpoch", "TrainAcc", "ValAcc", "EpochsRun"])

    for act in activations:
        logger.info(f"\n--- Training with activation: {act} ---")

        model = CNNClassifier(
            vocab_size=len(vocab),
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT,
            padding_idx=vocab["<PAD>"],
            activation=act
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_REG)

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
            logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

           
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
                logger.info("Validation accuracy improved. Resetting early stopping counter.")

                # Save best overall model so far
                if val_acc > overall_best_val_acc:
                    overall_best_val_acc = val_acc
                    overall_best_model = model.state_dict()
                    overall_best_act = act

            else:
                epochs_no_improve += 1
                logger.info(f"No improvement. Patience counter: {epochs_no_improve}/{PATIENCE}")
                if epochs_no_improve >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        writer.writerow([act, best_epoch, round(best_train_acc, 4), round(best_val_acc, 4), epoch + 1])
        logger.info(f"Finished {act} activation\n")

if overall_best_model:
    save_path = f"best_cnn_model_{overall_best_act}.pt"
    torch.save(overall_best_model, save_path)
    logger.info(f"Best model saved as {save_path} with val accuracy {overall_best_val_acc:.4f}")
