from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
from evaluate import load
import torch
import numpy as np
import pandas as pd
import os
from dataset import ClickbaitSpoilerDatasetParagraphLevel

# configs
base_checkpoint_dir = "./checkpoints/distilbart"
val_path = "data/val.jsonl"
batch_size = 4

# load tokenizer and val data
tokenizer = AutoTokenizer.from_pretrained(base_checkpoint_dir)
val_dataset = ClickbaitSpoilerDatasetParagraphLevel(val_path, tokenizer_name=base_checkpoint_dir)

# laod metrics
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")

# get checkpoints
checkpoints_dir = "./checkpoints/distilbart"
checkpoint_paths = sorted([
    os.path.join(checkpoints_dir, d)
    for d in os.listdir(checkpoints_dir)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoints_dir, d))
])


results = []

for path in checkpoint_paths:
    print(f"Evaluating {path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(path)

    # set up trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=4,
        # do_train=False,
        # do_eval=True,
        report_to="none",
        fp16=True,
        remove_unused_columns=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    # generate predictions
    preds = trainer.predict(val_dataset)

    print("Type:", type(preds.predictions))
    print("Shape:", np.shape(preds.predictions))
    print("Dtype:", preds.predictions.dtype)
    print("Min:", np.min(preds.predictions))
    print("Max:", np.max(preds.predictions))
    print("Sample:", preds.predictions[0][:10])
    print("Any -100 in predictions?", np.any(preds.predictions == -100))

    pred_ids = preds.predictions
    #     # Ensure predictions are valid integers
    # if isinstance(preds.predictions, tuple):
    #     pred_logits = preds.predictions[0]
    # else:
    #     pred_logits = preds.predictions

    # if pred_logits.ndim == 3:  # Likely logits
    #     pred_ids = np.argmax(pred_logits, axis=-1)
    # else:
    #     pred_ids = pred_logits

    # Replace -100 with pad token ID before decoding
    # pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)
    label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)

    # # Check for non-integer or invalid token values before decoding
    # if not np.issubdtype(pred_ids.dtype, np.integer):
    #     raise ValueError("Predictions contain non-integer values, likely due to FP16 overflow or incorrect decoding.")

    # # Optional: clip extreme values if necessary (just to recover partial outputs)
    # pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    # label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)

    # print("Sample prediction IDs:", pred_ids[0][:10])
    # print("Prediction shape:", pred_ids.shape)

    #diagnostic of overflow error
    print("pred_ids dtype:", pred_ids.dtype)
    if not np.issubdtype(pred_ids.dtype, np.integer):
      print("⚠️ pred_ids are not integers!")

    print("pred_ids min:", np.min(pred_ids))
    print("pred_ids max:", np.max(pred_ids))
    print("tokenizer vocab size:", tokenizer.vocab_size)

    if np.any(pred_ids < 0):
      print("⚠️ Negative values found in pred_ids")
      
    if np.any(pred_ids > tokenizer.vocab_size):
      print("⚠️ Values exceed tokenizer vocab size")

    if np.any(np.isnan(pred_ids)):
      print("⚠️ NaNs detected in pred_ids")

    if np.any(np.isinf(pred_ids)):
      print("⚠️ Infs detected in pred_ids")


    # decode
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # compute metrics
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

    # Save predictions
    df = pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs})
    df.to_csv(f"val_predictions_{os.path.basename(path)}.csv", index=False)
    print(f"Saved predictions to val_predictions_{os.path.basename(path)}.csv")

    # Log
    results.append({
        "checkpoint": path,
        "meteor": meteor_score,
        "bleu": bleu_score
    })

pd.DataFrame(results).to_csv("distilbart_eval_summary.csv", index=False)
print("\n Evaluation summary saved to distilbart_eval_summary.csv")
