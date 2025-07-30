from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from evaluate import load
import numpy as np
import pandas as pd
import os
from distilbart_dataset import ClickbaitSpoilerDatasetParagraphLevel


tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
val_dataset = ClickbaitSpoilerDatasetParagraphLevel("data/val.jsonl")

meteor = load("meteor")
bleu = load("bleu")

checkpoints_dir = "./checkpoints/distilbart"
checkpoint_paths = [os.path.join(checkpoints_dir, d) for d in sorted(os.listdir(checkpoints_dir)) if "checkpoint" in d]

results = []
for path in checkpoint_paths:
    print(f"Evaluating {path}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(path)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=4,
        do_train=False,
        do_eval=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer
    )

    preds = trainer.predict(val_dataset)
    
        # Ensure predictions are valid integers
    if isinstance(preds.predictions, tuple):
        raw_preds = preds.predictions[0]
    else:
        raw_preds = preds.predictions

    if raw_preds.ndim == 3:  # Likely logits
        pred_ids = np.argmax(raw_preds, axis=-1)
    else:
        pred_ids = raw_preds

    # Check for non-integer or invalid token values before decoding
    if not np.issubdtype(pred_ids.dtype, np.integer):
        raise ValueError("Predictions contain non-integer values, likely due to FP16 overflow or incorrect decoding.")

    # Optional: clip extreme values if necessary (just to recover partial outputs)
    pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    label_ids = np.where(preds.label_ids != -100, preds.label_ids, tokenizer.pad_token_id)

    print("Sample prediction IDs:", pred_ids[0][:10])
    print("Prediction shape:", pred_ids.shape)

    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_refs)["meteor"]
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

    # Save predictions
    df = pd.DataFrame({"generated": decoded_preds, "reference": decoded_refs})
    df.to_csv(f"val_predictions_{os.path.basename(path)}.csv", index=False)

    # Log
    results.append({
        "checkpoint": path,
        "meteor": meteor_score,
        "bleu": bleu_score
    })

pd.DataFrame(results).to_csv("distilbart_eval_summary.csv", index=False)

# import os
# import re
# import pandas as pd
# from transformers import AutoModelForSeq2SeqLM
# from train_distilbart import tokenizer, trainer, val_dataset, bleu, log_metrics

# output_dir = "checkpoints/distilbart"

# # Find all valid checkpoint folders
# checkpoints = sorted([
#     os.path.join(output_dir, d) for d in os.listdir(output_dir)
#     if re.match(r"^checkpoint-\d+$", d)
# ], key=lambda x: int(x.split("-")[-1]))

# best_meteor = 0
# best_checkpoint = None

# for i, checkpoint_path in enumerate(checkpoints, 1):
#     print(f"\n Evaluating checkpoint: {checkpoint_path}")
#     model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(trainer.args.device)
#     trainer.model = model

#     eval_result = trainer.evaluate()
#     val_loss = eval_result["eval_loss"]
#     val_meteor = eval_result["eval_meteor"]

#     preds = trainer.predict(val_dataset)
#     decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
#     decoded_refs = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
#     val_bleu = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

#     df = pd.DataFrame({
#         "index": list(range(len(decoded_preds))),
#         "generated": decoded_preds,
#         "reference": decoded_refs
#     })
#     df.to_csv(f"val_predictions_distilbart_epoch{i}.csv", index=False)

#     log_metrics(i, val_loss, val_meteor, val_bleu, 0)

#     if val_meteor > best_meteor:
#         best_meteor = val_meteor
#         best_checkpoint = checkpoint_path

# print(f"\n Best checkpoint: {best_checkpoint} with METEOR = {best_meteor:.4f}")
