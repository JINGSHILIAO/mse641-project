import torch
import pandas as pd
from scripts.task1.dataset import get_dataloader, reverse_label_map, build_vocab
from models.cnn_classifier_task1 import CNNClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

TEST_PATH = "data/test.jsonl"
TRAIN_PATH = "data/train.jsonl" 
MODEL_PATH = "best_cnn_model_sigmoid.pt"
SUBMIT_PATH = "task1_cnn_submission.csv"

BATCH_SIZE = 32
EMBED_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
NUM_CLASSES = 3

print("Building vocab...")
train_loader, vocab = get_dataloader(TRAIN_PATH, batch_size=1)  # we only need vocab
test_loader, _ = get_dataloader(TEST_PATH, vocab=vocab, batch_size=BATCH_SIZE, is_test=True)

print("Loading model...")
model = CNNClassifier(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    dropout=0.3,
    padding_idx=vocab["<PAD>"],
    activation="sigmoid"
)
model.load_state_dict(torch.load(MODEL_PATH))
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

print(f"Saving to {SUBMIT_PATH}...")
df = pd.DataFrame({
    "id": list(range(len(results))),
    "spoilerType": results
})
df.to_csv(SUBMIT_PATH, index=False)

print("Submission file created.")
