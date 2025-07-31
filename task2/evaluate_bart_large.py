import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from distilbart_dataset import ClickbaitSpoilerDatasetParagraphLevel
import evaluate

# --- Configuration ---
model_dir = "checkpoints/bart-large-final"
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
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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
print("📊 Final Evaluation on bart-large-first:")
print(metrics)

# --- Generate predictions and save to CSV ---
raw_preds = trainer.predict(eval_dataset)
pred_ids = np.argmax(raw_preds.predictions, axis=-1)
decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(raw_preds.label_ids, skip_special_tokens=True)

rows = []
for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
    rows.append({
        "id": idx,
        "prediction": pred.strip(),
        "reference": label.strip()
    })

df = pd.DataFrame(rows)
df.to_csv("bart_large_generated.csv", index=False)
print("✅ Predictions saved to bart_large_generated.csv")
