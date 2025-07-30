import time
import os
import torch
import pandas as pd
from transformers import (
#    BartForConditionalGeneration,
#    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
import evaluate
from distilbart_dataset import ClickbaitSpoilerDatasetParagraphLevel
# from nltk_setup import ensure_nltk_resources
import re

# --- Config ---
model_name = "sshleifer/distilbart-cnn-12-6"
train_path = "data/train.jsonl"
val_path = "data/val.jsonl"
output_dir = "./checkpoints/distilbart"
batch_size = 8
learning_rate = 5e-5
# num_epochs = 1
num_epochs = 3

# --- Load model and tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- Load datasets ---
train_dataset = ClickbaitSpoilerDatasetParagraphLevel(train_path, tokenizer_name=model_name)
val_dataset = ClickbaitSpoilerDatasetParagraphLevel(val_path, tokenizer_name=model_name)

# --- Metrics: METEOR, BLEU ---
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return meteor.compute(predictions=preds, references=refs)

def log_metrics(epoch, loss, meteor_score, bleu_score, train_time):
    from pathlib import Path
    file_path = "distilbart_val_metrics.csv"
    write_header = epoch == 1 and not Path(file_path).exists()
    with open(file_path, "a") as f:
        if write_header:
            f.write("epoch,train_time_sec,val_loss,meteor,bleu\n")
        f.write(f"{epoch},{train_time:.2f},{loss:.4f},{meteor_score:.4f},{bleu_score:.4f}\n")


# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_total_limit=3,
    remove_unused_columns=True,
    logging_first_step=True,
    report_to="none",
    fp16 = True, # Enable mixed precision for GPU
    disable_tqdm=False
)

# --- Trainer setup ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    compute_metrics=compute_metrics
)

# --- Train and time it ---
start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time

# --- Track Evaluation ---
# eval_result = trainer.evaluate()
best_meteor = 0
best_checkpoint = None

# List actual checkpoint folders
checkpoints = sorted([
    os.path.join(output_dir, d) for d in os.listdir(output_dir)
    if re.match(r"^checkpoint-\d+$", d)
], key=lambda x: int(x.split("-")[-1]))

# Evaluation loop
for epoch, checkpoint_path in enumerate(checkpoints, 1):
    print(f"\nEvaluating checkpoint: {checkpoint_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(trainer.args.device)
    trainer.model = model

    eval_result = trainer.evaluate()
    val_loss = eval_result["eval_loss"]
    val_meteor = eval_result["eval_meteor"]

    # Get BLEU
    preds = trainer.predict(val_dataset)
    decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
    val_bleu = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

    # Save prediction csv
    df = pd.DataFrame({
        "index": list(range(len(decoded_preds))),
        "generated": decoded_preds,
        "reference": decoded_refs
    })
    df.to_csv(f"val_predictions_distilbart_epoch{epoch}.csv", index=False)

    # Log metrics
    log_metrics(epoch, val_loss, val_meteor, val_bleu, elapsed)

    # Track best model
    if val_meteor > best_meteor:
        best_meteor = val_meteor
        best_checkpoint = checkpoint_path

print(f"\n Best checkpoint: {best_checkpoint} with METEOR = {best_meteor:.4f}")

# for epoch in range(1, num_epochs + 1):
#     print(f"\n=== Epoch {epoch} ===")
    
#     # Resume training for this epoch (or continue if inside single run)
#     start = time.time()
#     trainer.train(resume_from_checkpoint=None if epoch == 1 else f"{output_dir}/checkpoint-{epoch * len(train_dataset) // batch_size}")
#     elapsed = time.time() - start
    
#     # Evaluate
#     eval_result = trainer.evaluate()
#     val_loss = eval_result["eval_loss"]
#     val_meteor = eval_result["eval_meteor"]
    
#     # Compute BLEU manually
#     preds = trainer.predict(val_dataset)
#     decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
#     decoded_refs = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
#     val_bleu = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

#     # Log to console
#     print(f"Validation Loss: {val_loss:.4f}")
#     print(f"Validation METEOR: {val_meteor:.4f}")
#     print(f"Validation BLEU: {val_bleu:.4f}")
    
#     # Log to CSV
#     log_metrics(epoch, val_loss, val_meteor, val_bleu, elapsed)
    
#     # Save predictions for this epoch
#     pd.DataFrame({
#         "index": list(range(len(decoded_preds))),
#         "generated": decoded_preds,
#         "reference": decoded_refs
#     }).to_csv(f"val_predictions_distilbart_epoch{epoch}.csv", index=False)
    
#     # Track best checkpoint
#     if val_meteor > best_meteor:
#         best_meteor = val_meteor
#         best_checkpoint = f"{output_dir}/checkpoint-{epoch * len(train_dataset) // batch_size}"

# print("\n Final Evaluation Summary")
# print(f"Best METEOR: {best_meteor:.4f} at {best_checkpoint}")
# print("You can load this checkpoint later for test generation.")


# print("\nTraining Complete")
# print(f"Training time: {elapsed:.2f} seconds")
# print(f"Validation loss: {eval_result['eval_loss']:.4f}")
# print(f"Validation METEOR: {eval_result['eval_meteor']:.4f}")


# # --- Generate and save predictions ---
# predictions = trainer.predict(val_dataset)
# decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
# decoded_refs = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

# # Save results to CSV
# results_df = pd.DataFrame({
#     "index": list(range(len(decoded_preds))),
#     "generated": decoded_preds,
#     "reference": decoded_refs
# })
# results_df.to_csv("val_predictions_distilbart.csv", index=False)
# print(" Predictions saved to val_predictions_distilbart.csv")
# print(f" Checkpoints saved in: {output_dir}")
