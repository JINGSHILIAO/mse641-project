import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from distilbart_dataset import get_dataloaders  

# -------- Settings --------
test_path = "data/test.jsonl"
checkpoint_path = "./checkpoints/distilbart/checkpoint-1008"
output_path = "distilbart_submission.csv"

# -------- Load Model & Tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set generation configs
model.generation_config.num_beams = 1
model.generation_config.min_length = 1
model.generation_config.length_penalty = 1.0
model.generation_config.early_stopping = False

# -------- Load Test Data --------
test_data = get_dataloader(test_path)
test_ids = [ex["id"] for ex in test_data]
test_inputs = [ex["postText"][0] for ex in test_data]

# -------- Generate Predictions --------
inputs = tokenizer(test_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs)

preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# -------- Build Submission DataFrame --------
submission_df = pd.DataFrame({
    "id": test_ids,
    "spoiler": preds
})

# -------- Save --------
submission_df.to_csv(output_path, index=False)
print(f" Submission saved to {output_path}")
