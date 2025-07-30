import time
import pandas as pd
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate
from distilbart_dataset import ClickbaitSpoilerDatasetParagraphLevel

# --- Config ---
model_name = "sshleifer/distilbart-cnn-12-6"
train_path = "data/train.jsonl"
val_path = "data/val.jsonl"
output_dir = "./checkpoints/distilbart"
batch_size = 8
# num_epochs = 1
num_epochs = 3

# --- Load model and tokenizer ---
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# --- Load datasets ---
train_dataset = ClickbaitSpoilerDatasetParagraphLevel(train_path, tokenizer_name=model_name)
val_dataset = ClickbaitSpoilerDatasetParagraphLevel(val_path, tokenizer_name=model_name)

# --- Metric: METEOR ---
meteor = evaluate.load("meteor")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return meteor.compute(predictions=preds, references=refs)

# --- Training arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=5e-5,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_total_limit=2,
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
end_time = time.time()
elapsed = end_time - start_time

# --- Evaluate ---
eval_result = trainer.evaluate()
print("\nTraining Complete")
print(f"Training time: {elapsed:.2f} seconds")
print(f"Validation loss: {eval_result['eval_loss']:.4f}")
print(f"Validation METEOR: {eval_result['eval_meteor']:.4f}")

# --- Generate and save predictions ---
predictions = trainer.predict(val_dataset)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_refs = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

# Save results to CSV
results_df = pd.DataFrame({
    "index": list(range(len(decoded_preds))),
    "generated": decoded_preds,
    "reference": decoded_refs
})
results_df.to_csv("val_predictions_distilbart.csv", index=False)
print(" Predictions saved to val_predictions_distilbart.csv")
print(f" Checkpoints saved in: {output_dir}")
