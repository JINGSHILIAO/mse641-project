GENERATE SUBMISSION
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from load_test_inputs import load_test_inputs
from dataset import ClickbaitSpoilerDatasetParagraphLevel
from torch.utils.data import DataLoader


checkpoint_path = "./checkpoints/distilbart_baseline/checkpoint-504"
# checkpoint_path = "./checkpoints/bartlarge_baseline_continued/checkpoint-672"
test_path = "data/test.jsonl"
output_path = "bartlarge_504_best_decode.csv"
max_input_tokens = 1000  # token cutoff to match training logic
max_target_tokens = 64
max_paragraphs = 15
batch_size = 4 

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

test_dataset = ClickbaitSpoilerDatasetParagraphLevel(
    jsonl_path=test_path,
    tokenizer_name=checkpoint_path,
    max_input_tokens=max_input_tokens,
    max_target_tokens=max_target_tokens,
    max_paragraphs=max_paragraphs,
    is_test=True  # not expecting labels
)

# Set generation configs
# model.generation_config.num_beams = 1
# model.generation_config.min_length = 1
# model.generation_config.max_length = 64
# model.generation_config.length_penalty = 0.8
# model.generation_config.no_repeat_ngram_size = 3
# model.generation_config.early_stopping = False

# double check configs
# print(model.generation_config)

# ids, texts = load_test_inputs(test_path, tokenizer_name=checkpoint_path, max_tokens=max_tokens)

collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

outputs = []

for batch in test_loader:
    batch_inputs = {k: v.to(model.device) for k, v in batch.items() if k != "ids"}
    with torch.no_grad():
        batch_outputs = model.generate(
            **batch_inputs,
            num_beams=1,
            max_length=64,
            min_length=1,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
    outputs.extend(decoded)

# save to csv
test_ids = []
with open(test_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        test_ids.append(obj["uuid"])

assert len(test_ids) == len(outputs), "Mismatch between IDs and predictions"

submission_df = pd.DataFrame({"id": test_ids, "spoiler": outputs})
submission_df.to_csv(output_path, index=False)
print(f"Saved submission to {output_path}")

# for i in range(0, len(texts), batch_size):
#     batch_texts = texts[i:i + batch_size]
#     batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

#     with torch.no_grad():
#         batch_outputs = model.generate(**batch_inputs)

#     outputs.extend(batch_outputs)

# spoilers = tokenizer.batch_decode(outputs, skip_special_tokens=True)


# submission_df = pd.DataFrame({
#     "id": ids, 
#     "spoiler": spoilers
# })

# submission_df.to_csv(output_path, index=False)
# print(f" Submission saved to {output_path}")
