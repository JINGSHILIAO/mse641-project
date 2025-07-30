import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from distilbart_load_test_inputs import load_test_inputs

# -------- Configurations --------
checkpoint_path = "./checkpoints/bart-large/checkpoint-XXXX"  # <- update this to best checkpoint
test_path = "data/test.jsonl"
output_path = "bart_large_submission.csv"
max_tokens = 950

# -------- Load Model & Tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Beam search setup
model.generation_config.num_beams = 4
model.generation_config.length_penalty = 1.2
model.generation_config.min_length = 1
model.generation_config.early_stopping = True

# -------- Load Test Data --------
ids, texts = load_test_inputs(test_path, tokenizer_name=checkpoint_path, max_tokens=max_tokens)

# -------- Generate in Batches --------
batch_size = 4
outputs = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        batch_outputs = model.generate(**inputs)

    outputs.extend(batch_outputs)

spoilers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# -------- Save Submission --------
submission_df = pd.DataFrame({
    "id": ids,
    "spoiler": spoilers
})

submission_df.to_csv(output_path, index=False)
print(f"✅ Submission saved to {output_path}")
