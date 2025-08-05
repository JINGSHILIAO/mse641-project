
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import random
import numpy as np
import os
import sys
import csv
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.rnn_classifier import RNNClassifier
from scripts.task1.dataset import get_dataloader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

BATCH_SIZE = 32
EMBED_DIM = 100
HIDDEN_DIM = 128
LR = 0.001
DROPOUT = 0.3
NUM_CLASSES = 3
NUM_EPOCHS = 50
PATIENCE = 5

train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=BATCH_SIZE)
val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=BATCH_SIZE)


rnn_types = ["lstm", "bilstm"]
pooling_methods = ["last", "mean", "max"]
layer_options = [1, 2]

os.makedirs("experiments", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"experiments/rnn_gridsearch_{timestamp}.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["RNN_Type", "Pooling", "Num_Layers", "BestEpoch", "TrainAcc", "ValAcc", "EpochsRun", "TrainTime_sec"])

    for rnn_type in rnn_types:
        for pooling in pooling_methods:
            for num_layers in layer_options:
                logger.info(f"\n--- Training {rnn_type.upper()} | Pooling: {pooling.upper()} | Layers: {num_layers} ---")
                model = RNNClassifier(
                    vocab_size=len(vocab),
                    embed_dim=EMBED_DIM,
                    hidden_dim=HIDDEN_DIM,
                    num_classes=NUM_CLASSES,
                    rnn_type=rnn_type,
                    dropout=DROPOUT,
                    padding_idx=vocab["<PAD>"],
                    pooling=pooling,
                    num_layers=num_layers
                ).to(DEVICE)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LR)

                best_val_acc = 0.0
                best_train_acc = 0.0
                best_epoch = 0
                epochs_no_improve = 0
                start_time = time.time()

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
                        logger.info("Validation accuracy improved.")
                    else:
                        epochs_no_improve += 1
                        logger.info(f"No improvement. Patience counter: {epochs_no_improve}/{PATIENCE}")
                        if epochs_no_improve >= PATIENCE:
                            logger.info("Early stopping triggered.")
                            break

                train_time = round(time.time() - start_time, 2)
                writer.writerow([rnn_type, pooling, num_layers, best_epoch, round(best_train_acc, 4), round(best_val_acc, 4), epoch + 1, train_time])
                logger.info(f"Finished {rnn_type.upper()} | Pooling: {pooling.upper()} | Layers: {num_layers} | Best Val Acc: {best_val_acc:.4f} | Time: {train_time} sec\n")

logger.info(f"Experiment results saved to: {csv_filename}")
