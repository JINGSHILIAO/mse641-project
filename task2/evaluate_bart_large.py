import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from distilbart_dataset import ClickbaitSpoilerDatasetParagraphLevel
import evaluate

# --- Configuration ---
model_dir = "checkpoints/bart-large" # change this to the checkpoint dir 
eval_file = "data/val.jsonl"
model_name = "facebook/bart-large"
batch_size = 4

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# --- Load evaluation dataset ---
eval_dataset = ClickbaitSpoilerDatasetParagraphLevel(eval_file, model_name)

# --- Load official evaluation metrics ---
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pred_ids = np.argmax(predictions, axis=-1)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    cleaned_labels = [
    [token if token != -100 else tokenizer.pad_token_id for token in label]
    for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "meteor": meteor_result["meteor"],
        "bleu": bleu_result["bleu"]
    }

# --- Seq2Seq Trainer Setup ---
args = Seq2SeqTrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=batch_size,
    do_predict=True,
    predict_with_generate=True,
    logging_dir="./logs",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics=compute_metrics,
    eval_dataset=eval_dataset,
)

# --- Evaluate ---
metrics = trainer.evaluate()
print(" Evaluation on {model_dir}:")
print(metrics)

# --- Generate predictions using model.generate() ---
model.eval()
generated_texts = []
reference_texts = []

for example in eval_dataset:
    input_ids = example["input_ids"].unsqueeze(0).to(model.device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    decoded_pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    label_ids = example["labels"]
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    decoded_label = tokenizer.decode(label_ids, skip_special_tokens=True)

    generated_texts.append(decoded_pred.strip())
    reference_texts.append(decoded_label.strip())

# --- Save to CSV ---
rows = []
for idx, (pred, label) in enumerate(zip(generated_texts, reference_texts)):
    rows.append({
        "id": idx,
        "prediction": pred,
        "reference": label
    })

df = pd.DataFrame(rows)
df.to_csv("bart_large_generated.csv", index=False)
print("Predictions saved to bart_large_generated.csv")

# # --- Generate predictions and save to CSV ---
# raw_preds = trainer.predict(eval_dataset)
# pred_ids = np.argmax(raw_preds.predictions, axis=-1)
# decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

# cleaned_label_ids = [
#     [token if token != -100 else tokenizer.pad_token_id for token in label]
#     for label in raw_preds.label_ids
# ]
# decoded_labels = tokenizer.batch_decode(cleaned_label_ids, skip_special_tokens=True)


# rows = []
# for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
#     rows.append({
#         "id": idx,
#         "prediction": pred.strip(),
#         "reference": label.strip()
#     })

# df = pd.DataFrame(rows)
# df.to_csv("bart_large_generated.csv", index=False)
# print("Predictions saved to bart_large_generated.csv")
