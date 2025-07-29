
import torch
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from models.transformer_classifier import TransformerClassifier
from transformer_based_dataset import get_dataloader

# --- Settings ---
# model_path = "task1/outputs/best_bert_base_uncased_model.pt"
# model_name = "bert-base-uncased"
model_path = "task1/outputs/best_distilbert_model.pt"
model_name = "distilbert-base-uncased"
test_path = "data/test.jsonl"

output_csv = "submission_task1_distilbert.csv"
batch_size = 32
max_len = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Label Map ---
id2label = {0: "phrase", 1: "passage", 2: "multi"}

# --- Load test data ---
test_loader = get_dataloader(test_path, batch_size=batch_size, tokenizer_name=model_name, max_len=max_len, is_test=True)

# --- Load model ---
model = TransformerClassifier(model_name=model_name, num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Predict ---
all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

# --- Load IDs from test.jsonl ---
ids = []
with open(test_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        ids.append(entry["id"])

# --- Save to CSV ---
df = pd.DataFrame({
    "id": ids,
    "spoilerType": [id2label[p] for p in all_preds]
})
df.to_csv(output_csv, index=False)
print(f"Submission saved to: {output_csv}")
