import time
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
#    BartForConditionalGeneration,
#    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import evaluate
from evaluate import load
from dataset import ClickbaitSpoilerDatasetParagraphLevel
from transformers.trainer_callback import EarlyStoppingCallback

# configs
model_name = "sshleifer/distilbart-cnn-12-6"
train_path = "data/train.jsonl"
val_path = "data/val.jsonl"
output_dir = "./checkpoints/distilbart"
batch_size = 4
num_epochs = 5

# more configs to tune and help model trainnig
# weight_decay=0.01
warmup_steps=500
label_smoothing_factor=0.1 # help improve generalization
max_grad_norm=1.0
learning_rate = 2e-5

max_input_tokens = 1000
max_target_tokens = 64
max_paragraphs = 15

# load tokenizer, model and dataset
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
train_dataset = ClickbaitSpoilerDatasetParagraphLevel(jsonl_path=train_path, tokenizer_name=model_name, max_input_tokens=max_input_tokens, max_target_tokens=max_target_tokens, max_paragraphs = max_paragraphs)
val_dataset = ClickbaitSpoilerDatasetParagraphLevel(jsonl_path=val_path, tokenizer_name=model_name, max_input_tokens=max_input_tokens, max_target_tokens=max_target_tokens, max_paragraphs = max_paragraphs)

# define metrics
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

def compute_metrics(pred):
    label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]

    return {"meteor": meteor_score, "bleu": bleu_score}

# training setup
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4, # to simulate larger batch size (16) for stability
    warmup_steps=warmup_steps,
    label_smoothing_factor=label_smoothing_factor,
    max_grad_norm=max_grad_norm, # helps stablize training 
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    # weight_decay=weight_decay,
    predict_with_generate=True, # need evaluation in eval scripts
    # save_total_limit=2,
    report_to="none",
    fp16=True,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="meteor",
    greater_is_better=True
)

# debug for the tensor mismatch problem
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100 
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


# Train and time it
start_train = time.time()
trainer.train()
train_time = time.time() - start_train
print(f"\n Total training time: {train_time:.2f} seconds")

trainer.save_model(output_dir)
print(f"Model saved to: {output_dir}")


# # ---------------- Evaluation ----------------
# meteor = load("meteor")
# bleu = load("bleu")

checkpoints = sorted([
    os.path.join(output_dir, d)
    for d in os.listdir(output_dir)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
])

results = []

for cp in checkpoints:
    print(f"\n Evaluating {cp}")
    model = AutoModelForSeq2SeqLM.from_pretrained(cp)
    
    # Eval configs
    model.generation_config.num_beams = 4
    model.generation_config.min_length = 1 # avoid empty outputs
    model.generation_config.max_length = max_target_tokens
    model.generation_config.length_penalty = 0.9 # slight prefrence for shorter output
    model.generation_config.early_stopping = True

#     # debug: check config
    print(model.generation_config)

    trainer.model = model

#     start_pred = time.time()
    trainer.model.to(trainer.args.device)
    with torch.no_grad():
      preds = trainer.predict(val_dataset)

#     pred_time = time.time() - start_pred

#     raw_preds = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
#     if raw_preds.ndim == 3:
#         pred_ids = np.argmax(raw_preds, axis=-1)
#     else:
#         pred_ids = raw_preds
#     pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Save predictions
    csv_path = f"val_predictions_{os.path.basename(cp)}.csv"
    pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs}).to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

    # Save summary
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]
    results.append({
        "checkpoint": os.path.basename(cp),
        "meteor": meteor_score,
        "bleu": bleu_score
    })

pd.DataFrame(results).to_csv("distilbart_eval_summary.csv", index=False)
print("Saved eval summary to distilbart_eval_summary.csv")

#     # Save predictions
#     cp_name = os.path.basename(cp)
#     df = pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs})
#     df.to_csv(f"val_predictions_{cp_name}.csv", index=False)

#     # Compute metrics
#     meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
#     bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

#     results.append({
#         "checkpoint": cp_name,
#         "meteor": meteor_score,
#         "bleu": bleu_score,
#         "train_time_sec": round(train_time, 2) if cp == checkpoints[-1] else "",
#         "predict_time_sec": round(pred_time, 2)
#     })

# # Save metric summary
# summary_df = pd.DataFrame(results)
# summary_df.to_csv("distilbart_eval_summary.csv", index=False)
# print("\n Evaluation complete. Results saved to distilbart_eval_summary.csv")

# Ensure generation tokens are configured
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.decoder_start_token_id = tokenizer.bos_token_id


# # --- Metrics: METEOR, BLEU ---
# meteor = evaluate.load("meteor")
# bleu = evaluate.load("bleu")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred

#     if isinstance(predictions, tuple) or predictions.ndim == 3:
#         predictions = np.argmax(predictions, axis=-1)

#     predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

#     labels = np.where(labels != -100, tokenizer.pad_token_id, labels)
#     references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     return meteor.compute(predictions=predictions, references=references)

# def log_metrics(epoch, loss, meteor_score, bleu_score, train_time):
#     from pathlib import Path
#     file_path = "distilbart_val_metrics.csv"
#     write_header = epoch == 1 and not Path(file_path).exists()
#     with open(file_path, "a") as f:
#         if write_header:
#             f.write("epoch,train_time_sec,val_loss,meteor,bleu\n")
#         f.write(f"{epoch},{train_time:.2f},{loss:.4f},{meteor_score:.4f},{bleu_score:.4f}\n")


# # --- Training arguments ---
# training_args = Seq2SeqTrainingArguments(
#     output_dir=output_dir,
#     save_strategy="epoch",
#     logging_strategy="steps",
#     logging_steps=50,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     learning_rate=learning_rate,
#     num_train_epochs=num_epochs,
#     predict_with_generate=True,
#     generation_max_length=64,
#     generation_num_beams=2,  # one epoch experiment defaulted to 1
#     save_total_limit=2,
#     remove_unused_columns=True,
#     logging_first_step=True,
#     report_to="none",
#     fp16 = True, # Enable mixed precision for GPU
#     disable_tqdm=False
# )

# # --- Trainer setup ---
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
#     data_collator=DataCollatorForSeq2Seq(tokenizer),
#     compute_metrics=compute_metrics
# )

# # --- Train and time it ---
# start_time = time.time()
# train_result = trainer.train()
# elapsed = time.time() - start_time

# # --- Track Evaluation ---
# # eval_result = trainer.evaluate()
# best_meteor = 0
# best_checkpoint = None

# # List actual checkpoint folders
# checkpoints = sorted([
#     os.path.join(output_dir, d) for d in os.listdir(output_dir)
#     if re.match(r"^checkpoint-\d+$", d)
# ], key=lambda x: int(x.split("-")[-1]))

# # Evaluation loop
# for epoch, checkpoint_path in enumerate(checkpoints, 1):
#     print(f"\nEvaluating checkpoint: {checkpoint_path}")
#     model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(trainer.args.device)
#     trainer.model = model

#     eval_result = trainer.evaluate()
#     val_loss = eval_result["eval_loss"]
#     val_meteor = eval_result["eval_meteor"]

#     # Get BLEU
#     preds = trainer.predict(val_dataset)
#     decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
#     decoded_labels = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)
#     decoded_refs = tokenizer.batch_decode(decoded_labels, skip_special_tokens=True)
#     val_bleu = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

#     # Save prediction csv
#     df = pd.DataFrame({
#         "index": list(range(len(decoded_preds))),
#         "generated": decoded_preds,
#         "reference": decoded_refs
#     })
#     df.to_csv(f"val_predictions_distilbart_epoch{epoch}.csv", index=False)

#     # Log metrics
#     log_metrics(epoch, val_loss, val_meteor, val_bleu, elapsed)

#     # Track best model
#     if val_meteor > best_meteor:
#         best_meteor = val_meteor
#         best_checkpoint = checkpoint_path

# print(f"\n Best checkpoint: {best_checkpoint} with METEOR = {best_meteor:.4f}")


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
