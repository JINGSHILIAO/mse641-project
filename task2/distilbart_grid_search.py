import os
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from dataset import ClickbaitSpoilerDatasetParagraphLevel
import evaluate

# ---- CONFIGS ----
checkpoint_dir = "./distilbart-best-model/checkpoints/distilbart/checkpoint-504"  # best checkpoint
val_path = "data/val.jsonl"
model_name = "sshleifer/distilbart-cnn-12-6"
output_dir = "./grid_outputs"
os.makedirs(output_dir, exist_ok=True)

max_input_tokens = 1000
max_target_tokens = 64
max_paragraphs = 15
batch_size = 4

# ---- Load tokenizer, dataset, model ----
tokenizer = AutoTokenizer.from_pretrained(model_name)
val_dataset = ClickbaitSpoilerDatasetParagraphLevel(
    jsonl_path=val_path,
    tokenizer_name=model_name,
    max_input_tokens=max_input_tokens,
    max_target_tokens=max_target_tokens,
    max_paragraphs=max_paragraphs
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)

# ---- Setup Trainer ----
training_args = Seq2SeqTrainingArguments(
    output_dir="./tmp_eval",
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    report_to="none",
    fp16=True
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
)

# ---- Load Metrics ----
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

# ---- Search Space ----
beam_widths = [1, 4, 6]
length_penalties = [0.8, 1.0, 1.2]
max_lengths = [48, 64]
ngram_sizes = [2, 3]
search_grid = list(itertools.product(beam_widths, length_penalties, max_lengths, ngram_sizes))

results = []

# ---- Grid Search Loop ----
for num_beams, length_penalty, max_len, no_repeat_ngram_size in search_grid:
    print(f"\n[Config] Beams={num_beams}, LP={length_penalty}, MaxLen={max_len}, Ngram={no_repeat_ngram_size}")

    # Set generation config
    model.generation_config.num_beams = num_beams
    model.generation_config.length_penalty = length_penalty
    model.generation_config.max_length = max_len
    model.generation_config.min_length = 1
    model.generation_config.no_repeat_ngram_size = no_repeat_ngram_size
    model.generation_config.early_stopping = True

    print(model.generation_config)

    # Predict
    model.to(trainer.args.device)
    with torch.no_grad():
        preds = trainer.predict(val_dataset)

    # Decode
    pred_ids = preds.predictions
    label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Evaluate
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

    # Save predictions
    run_name = f"b{num_beams}_lp{length_penalty}_max{max_len}_ng{no_repeat_ngram_size}"
    pred_file = os.path.join(output_dir, f"val_preds_{run_name}.csv")
    pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs}).to_csv(pred_file, index=False)
    print(f"Saved to {pred_file} | METEOR={meteor_score:.4f}, BLEU={bleu_score:.4f}")

    # Track results
    results.append({
        "num_beams": num_beams,
        "length_penalty": length_penalty,
        "max_length": max_len,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "meteor": meteor_score,
        "bleu": bleu_score,
        "pred_file": os.path.basename(pred_file)
    })

# ---- Save Summary ----
results_df = pd.DataFrame(results).sort_values(by="meteor", ascending=False)
summary_file = os.path.join(output_dir, "inference_grid_results.csv")
results_df.to_csv(summary_file, index=False)
print(f"\nSaved full summary to {summary_file}")

