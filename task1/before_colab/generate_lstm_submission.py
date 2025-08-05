
import torch
import csv
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.rnn_classifier_v2 import RNNClassifier
from scripts.task1.dataset_v2 import get_dataloader, reverse_label_map

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

print("Building vocab...")
_, vocab = get_dataloader("data/train.jsonl", batch_size=1)
test_loader, _ = get_dataloader("data/test.jsonl", vocab=vocab, batch_size=32, is_test=True)

print("Loading model...")
model = RNNClassifier(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=256,
    num_classes=3,
    num_layers=1,
    bidirectional=False,
    dropout=0.6, 
    padding_idx=vocab["<PAD>"]
).to(DEVICE)

model.load_state_dict(torch.load("models/best_lstm_model_v2.pt", map_location=DEVICE))
model.eval()

print("Running predictions...")
results = []

with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        for pred in preds:
            label = reverse_label_map[pred.item()]
            results.append(label)

output_path = "results/task1_lstm_v2_submission.csv"
print(f"Saving to {output_path}...")

df = pd.DataFrame({
    "id": list(range(len(results))),
    "spoilerType": results
})
df.to_csv(output_path, index=False)

print("Submission file created.")
