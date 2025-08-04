import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from load_test_inputs import load_test_inputs
from dataset import ClickbaitSpoilerDatasetParagraphLevel
from torch.utils.data import DataLoader


# checkpoint_path = "./bartlarge_baseline_checkpoint_504/checkpoints/bartlarge_baseline/checkpoint-504"
# checkpoint_path = "./checkpoints/bartlarge_baseline_continued/checkpoint-672"
checkpoint_path = "./distilbart-checkpoint-504"
test_path = "data/test.jsonl"
output_path = "distilbart-checkpoint-504-configs-from-grid.csv"
max_input_tokens = 1000  # token cutoff to match training logic
max_target_tokens = 64
max_paragraphs = 15
batch_size = 4 

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.generation_config.num_beams = 6
model.generation_config.length_penalty = 0.8
model.generation_config.max_length = 64
model.generation_config.min_length = 1
model.generation_config.no_repeat_ngram_size = 2
model.generation_config.early_stopping = True


ids, texts = load_test_inputs(test_path, tokenizer_name=checkpoint_path, max_tokens=max_input_tokens)

outputs = []


for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    # batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

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
print(model.generation_config) # check config