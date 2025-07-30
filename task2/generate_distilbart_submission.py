import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from distilbart_load_test_inputs import load_test_inputs

checkpoint_path = "./checkpoints/distilbart/checkpoint-1008"
test_path = "data/test.jsonl"
output_path = "distilbart_submission.csv"
max_tokens = 950  # token cutoff to match training logic

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model configs (make sure it's the same)
model.generation_config.num_beams = 1
model.generation_config.min_length = 1
model.generation_config.length_penalty = 1.0
model.generation_config.early_stopping = False

ids, texts = load_test_inputs(test_path, tokenizer_name=checkpoint_path, max_tokens=max_tokens)

# inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

# # Generate predictions
# with torch.no_grad():
#     outputs = model.generate(**inputs)

# spoilers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

batch_size = 4 # CUDA OOM lol trying this instead
outputs = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        batch_outputs = model.generate(**batch_inputs)

    outputs.extend(batch_outputs)

spoilers = tokenizer.batch_decode(outputs, skip_special_tokens=True)


submission_df = pd.DataFrame({
    "id": ids, 
    "spoiler": spoilers
})

submission_df.to_csv(output_path, index=False)
print(f" Submission saved to {output_path}")