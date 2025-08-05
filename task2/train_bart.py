import os
import time
import torch
import argparse
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_callback import EarlyStoppingCallback
from dataset import ClickbaitSpoilerDatasetParagraphLevel 
import evaluate

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="sshleifer/distilbart-cnn-12-6")
parser.add_argument("--output_dir", type=str, default="./checkpoints/distilbart")
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--train_path", type=str, default="data/train.jsonl")
parser.add_argument("--val_path", type=str, default="data/val.jsonl")

# Training control
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--max_input_tokens", type=int, default=1000)
parser.add_argument("--max_target_tokens", type=int, default=64)
parser.add_argument("--max_paragraphs", type=int, default=15)

args = parser.parse_args()

# Model settings
if "bart-large" in args.model_name:
    learning_rate = 2e-5
    batch_size = 4
    gradient_accum_steps = 4
else:  # DistilBART
    learning_rate = 5e-5
    batch_size = 4
    gradient_accum_steps = 4

# Training configurations
warmup_steps = 200
label_smoothing = 0.1
max_grad_norm = 1.0

# Load tokenizer, model and datasets
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if args.resume_from_checkpoint:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_from_checkpoint)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

train_dataset = ClickbaitSpoilerDatasetParagraphLevel(
    jsonl_path=args.train_path,
    tokenizer_name=args.model_name,
    max_input_tokens=args.max_input_tokens,
    max_target_tokens=args.max_target_tokens,
    max_paragraphs=args.max_paragraphs
)

val_dataset = ClickbaitSpoilerDatasetParagraphLevel(
    jsonl_path=args.val_path,
    tokenizer_name=args.model_name,
    max_input_tokens=args.max_input_tokens,
    max_target_tokens=args.max_target_tokens,
    max_paragraphs=args.max_paragraphs
)

# Load metrics
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# def compute_metrics(pred):
#     label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

#     return {
#         "meteor": meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"],
#         "bleu": bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"],
#         "rougeL": rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
#     }

# Updated compute metrics method
# Added guards to prevent overflow
def compute_metrics(eval_preds):
    predictions, labels = eval_preds

    # Handle tuple case
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits to token IDs if needed
    if predictions.ndim == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions

    # clip out-of-range IDs and cast to int
    pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1).astype(np.int32)

    # Handle padding in labels
    cleaned_labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in labels
    ]
    cleaned_labels = np.array(cleaned_labels).astype(np.int32)

    # Decode both preds and refs
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)

    # Clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute metrics
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rougeL_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "meteor": meteor_result["meteor"],
        "bleu": bleu_result["bleu"],
        "rougeL": rougeL_result["rougeL"]
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accum_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    label_smoothing_factor=label_smoothing,
    max_grad_norm=max_grad_norm,
    predict_with_generate=True,
    logging_steps=100,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="meteor",
    greater_is_better=True,
    fp16=True
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100  # ignore pad tokens during loss computation
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    data_collator=data_collator
)

# to resume training from a checkpoint
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

# Train and time it
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"\n Total training time: {train_time:.2f} seconds")

trainer.save_model(args.output_dir)
print(f"Model saved to: {args.output_dir}")

# Evalaute checkpoints
checkpoints = sorted([
    os.path.join(args.output_dir, d)
    for d in os.listdir(args.output_dir)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))
])

results = []
for cp in checkpoints:
    print(f"\n Evaluating {cp}")
    model = AutoModelForSeq2SeqLM.from_pretrained(cp)
    model.to(trainer.args.device)

    # Setup generation parameters
    model.generation_config.num_beams = 4
    model.generation_config.min_length = 1
    model.generation_config.max_length = args.max_target_tokens
    model.generation_config.length_penalty = 1
    model.generation_config.early_stopping = True

    # confrim model generation configs
    print(model.generation_config)

    # Run prediction
    trainer.model = model
    with torch.no_grad():
        preds = trainer.predict(val_dataset)

    label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Save raw predictions
    pred_csv = f"val_predictions_{model.config._name_or_path.replace('/', '_')}_{os.path.basename(cp)}.csv"
    pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs}).to_csv(pred_csv, index=False)
    # print(f"Saved predictions to {pred_csv}")

    # Save metrics
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]
    rougeL_score = rouge.compute(predictions=decoded_preds, references=decoded_refs)["rougeL"]

    results.append({
        "checkpoint": os.path.basename(cp),
        "meteor": meteor_score,
        "bleu": bleu_score,
        "rougeL": rougeL_score
    })

# Save summary
summary_csv = f"{args.model_name.split('/')[-1]}_eval_summary.csv"
pd.DataFrame(results).to_csv(summary_csv, index=False)