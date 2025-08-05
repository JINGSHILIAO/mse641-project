
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from models.rnn_classifier_v2 import RNNClassifier
from dataset_v2 import get_dataloader

# Base config
base_config = {
    'embed_dim': 100,
    'hidden_dim': 128,
    'dropout': 0.5,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'l2_reg': 1e-5,
    'epochs': 10,
    'max_len': 200
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, vocab = get_dataloader("data/train.jsonl", batch_size=base_config['batch_size'], max_len=base_config['max_len'])
val_loader, _ = get_dataloader("data/val.jsonl", vocab=vocab, batch_size=base_config['batch_size'], max_len=base_config['max_len'])

for is_bi in [False, True]:
    tag = "BiLSTM" if is_bi else "LSTM"
    print(f"Training {tag}...")

    config = base_config.copy()
    config['model'] = tag
    config['bidirectional'] = is_bi

    model = RNNClassifier(
        vocab_size=len(vocab),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        num_layers=1,
        bidirectional=config['bidirectional']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg'])

    start_time = time.time()
    history = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"[{tag}] Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

    elapsed = round(time.time() - start_time, 2)
    summary = pd.DataFrame([{
        "Model": config['model'],
        "Embedding Dim": config['embed_dim'],
        "Hidden Dim": config['hidden_dim'],
        "Bidirectional": config['bidirectional'],
        "Dropout": config['dropout'],
        "Learning Rate": config['learning_rate'],
        "Batch Size": config['batch_size'],
        "L2 Reg": config['l2_reg'],
        "Epochs": config['epochs'],
        "Train Time (s)": elapsed,
        "Final Train Acc": round(train_acc, 4),
        "Final Val Acc": round(val_acc, 4)
    }])

    summary.to_csv(f"{tag.lower()}_results.csv", index=False)
