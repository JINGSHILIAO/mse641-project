import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ClickbaitSpoilerTypeDataset

# === CONFIG ===
MODEL_DIR = "roberta_base_baseline_3_epochs"
TEST_PATH = "data/test.jsonl"
OUTPUT_CSV = "roberta_baseline_submission.csv"
BATCH_SIZE = 4
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LABEL MAPPING ===
id2label = {0: "phrase", 1: "passage", 2: "multi"}

# === Load tokenizer and model ===
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# === Load test data ===
test_dataset = ClickbaitSpoilerTypeDataset(TEST_PATH, tokenizer_name="roberta-base", max_len=MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === Inference ===
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)

# === Convert predictions to labels
predicted_tags = [id2label[p] for p in all_preds]

# === Create submission DataFrame
submission_df = pd.DataFrame({
    "id": list(range(len(predicted_tags))),
    "spoilerType": predicted_tags
})
submission_df.to_csv(OUTPUT_CSV, index=False)

print(f"Submission CSV saved to {OUTPUT_CSV}")
