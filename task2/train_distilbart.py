import time
import os
import torch
import pandas as pd
import numpy as np
from transformers import (
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
