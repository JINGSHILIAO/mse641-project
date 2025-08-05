import os
import torch
# import nltk
import time
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
# from distilbart_dataset import get_dataloaders
from dataset import ClickbaitSpoilerDatasetParagraphLevel
# from nltk.translate.meteor_score import meteor_score
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("punkt_tab")

# configs
model_name = "facebook/bart-large"
output_dir = "./checkpoints/bart-large"
train_path = "data/train.jsonl"
# val_path = "data/val.jsonl"
num_epochs = 3 # <- change this to desired number of epochs
batch_size = 4
#learning_rate = 5e-5
learning_rate = 2e-5 # <- better LR for the dataset size
# eval_log_path = "bart_large_eval_summary.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# train_loader, val_loader = get_dataloaders(train_path, val_path, model_name, batch_size=per_device_batch_size)

# load dataset
train_dataset = ClickbaitSpoilerDatasetParagraphLevel(train_path, model_name)
# val_dataset = ClickbaitSpoilerDatasetParagraphLevel(val_path, model_name)


# define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    # eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    # eval_strategy = "no",
    weight_decay=0.01,
    # save_total_limit=3,
    generation_max_length=128,
    generation_num_beams=4,
    num_train_epochs=num_epochs,
    # predict_with_generate=True,
    logging_dir="./logs",
    report_to="none"
)


# smoothie = SmoothingFunction().method4

# # -------- Metric Function --------
# def compute_metrics(eval_preds):
#     predictions, labels = eval_preds

#     # Convert logits to token IDs (argmax over vocab dimension)
#     if isinstance(predictions, tuple):  # for models with extra outputs
#         predictions = predictions[0]

#     pred_ids = np.argmax(predictions, axis=-1)

#     # Decode
#     decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Compute metrics (example: METEOR and BLEU)
#     meteor_scores = [
#       meteor_score([ref.split()], pred.split())
#       for pred, ref in zip(decoded_preds, decoded_labels)
#     ]

#     bleu_scores = [
#         sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
#         for pred, ref in zip(decoded_preds, decoded_labels)
#     ]

#     return {
#         "meteor": sum(meteor_scores) / len(meteor_scores),
#         "bleu": sum(bleu_scores) / len(bleu_scores),
#     }

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    # compute_metrics=compute_metrics
)

# Train and track rain time
start_time = time.time()
# train_result = trainer.train()
trainer.train()
end_time = time.time()

train_time = round(end_time - start_time, 2)
print(f"\n Total training time: {train_time} seconds")

# save model
trainer.save_model(output_dir)
print(f" Model saved to {output_dir}")

# # -------- Evaluate All Checkpoints --------
# log_rows = []
# checkpoints = sorted([
#     os.path.join(output_dir, d) for d in os.listdir(output_dir)
#     if d.startswith("checkpoint-")
# ])

# print()
# for ckpt in checkpoints:
#     print(f"Evaluating {ckpt}")
#     model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(trainer.args.device)
#     trainer.model = model
#     eval_result = trainer.evaluate()
    
#     log_rows.append({
#         "checkpoint": ckpt,
#         "val_loss": eval_result.get("eval_loss"),
#         "meteor": eval_result.get("eval_meteor"),
#         "bleu": eval_result.get("eval_bleu"),
#         "train_time": train_time
#     })

# # -------- Save Log --------
# log_df = pd.DataFrame(log_rows)
# log_df.to_csv(eval_log_path, index=False)
# print(f"\n Evaluation complete. Results saved to {eval_log_path}")