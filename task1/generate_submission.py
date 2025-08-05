import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ClickbaitSpoilerTypeDataset

# MODEL_DIR = "roberta_base_baseline_3_epochs"
MODEL_DIR = "roberta_gridsearch_runs/roberta_d02_wd001"
TEST_PATH = "data/test.jsonl"
OUTPUT_CSV = "roberta_d02_wd001_submission.csv"
BATCH_SIZE = 4
MAX_LEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "phrase", 1: "passage", 2: "multi"}

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

test_dataset = ClickbaitSpoilerTypeDataset(TEST_PATH, tokenizer_name="roberta-base", max_len=MAX_LEN, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)

predicted_tags = [id2label[p] for p in all_preds]

submission_df = pd.DataFrame({
    "id": list(range(len(predicted_tags))),
    "spoilerType": predicted_tags
})
submission_df.to_csv(OUTPUT_CSV, index=False)


