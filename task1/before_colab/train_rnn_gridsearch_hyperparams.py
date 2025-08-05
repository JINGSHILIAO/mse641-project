
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

hidden_dims = [64, 128, 256]
dropouts = [0.1, 0.3, 0.5]
learning_rates = [1e-3, 5e-4, 1e-4]
batch_sizes = [32, 64]

RNN_TYPE = "lstm"
POOLING = "max"
NUM_LAYERS = 1
NUM_CLASSES = 3
EMBED_DIM = 100
NUM_EPOCHS = 50
PATIENCE = 5

os.makedirs("experiments", exist_ok=True)
os.makedirs("models", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"experiments/rnn_hyperparam_search_bestonly_{timestamp}.csv"
best_model_path = "models/best_lstm_model.pt"


best_overall_val_acc = 0.0
best_model_config = None
best_model_state_dict = None

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["HiddenDim", "Dropout", "LR", "BatchSize", "BestEpoch", "TrainAcc", "ValAcc", "EpochsRun", "TrainTime_sec"])

    for hidden_dim in hidden_dims:
        for dropout in dropouts:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    logger.info(f"\nRunning config: Hidden={hidden_dim}, Dropout={dropout}, LR={lr}, Batch={batch_size}")

                    train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=batch_size)
                    val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=batch_size)

                    model = RNNClassifier(
                        vocab_size=len(vocab),
                        embed_dim=EMBED_DIM,
                        hidden_dim=hidden_dim,
                        num_classes=NUM_CLASSES,
                        rnn_type=RNN_TYPE,
                        pooling=POOLING,
                        num_layers=NUM_LAYERS,
                        dropout=dropout,
                        padding_idx=vocab["<PAD>"]
                    ).to(DEVICE)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

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
                        else:
                            epochs_no_improve += 1
                            logger.info(f"No improvement. Patience counter: {epochs_no_improve}/{PATIENCE}")
                            if epochs_no_improve >= PATIENCE:
                                logger.info("Early stopping triggered.")
                                break

                    train_time = round(time.time() - start_time, 2)
                    writer.writerow([hidden_dim, dropout, lr, batch_size, best_epoch, round(best_train_acc, 4), round(best_val_acc, 4), epoch + 1, train_time])
                    logger.info(f"Finished: Best Val Acc = {best_val_acc:.4f} | Time: {train_time}s")

                    if best_val_acc > best_overall_val_acc:
                        best_overall_val_acc = best_val_acc
                        best_model_config = (hidden_dim, dropout, lr, batch_size)
                        best_model_state_dict = model.state_dict().copy()
                        logger.info(f"New best model found with val acc {best_val_acc:.4f}")

# Save the best model at the end
if best_model_state_dict:
    torch.save(best_model_state_dict, best_model_path)
    logger.info(f"Best model saved to {best_model_path} with val acc {best_overall_val_acc:.4f}")