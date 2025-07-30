import os
import re
import pandas as pd
from transformers import AutoModelForSeq2SeqLM
from train_distilbart import tokenizer, trainer, val_dataset, bleu, log_metrics

output_dir = "checkpoints/distilbart"

# Find all valid checkpoint folders
checkpoints = sorted([
    os.path.join(output_dir, d) for d in os.listdir(output_dir)
    if re.match(r"^checkpoint-\d+$", d)
], key=lambda x: int(x.split("-")[-1]))

best_meteor = 0
best_checkpoint = None

for i, checkpoint_path in enumerate(checkpoints, 1):
    print(f"\n🔍 Evaluating checkpoint: {checkpoint_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    trainer.model = model

    eval_result = trainer.evaluate()
    val_loss = eval_result["eval_loss"]
    val_meteor = eval_result["eval_meteor"]

    preds = trainer.predict(val_dataset)
    decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)
    val_bleu = bleu.compute(predictions=decoded_preds, references=decoded_refs)["bleu"]

    df = pd.DataFrame({
        "index": list(range(len(decoded_preds))),
        "generated": decoded_preds,
        "reference": decoded_refs
    })
    df.to_csv(f"val_predictions_distilbart_epoch{i}.csv", index=False)

    log_metrics(i, val_loss, val_meteor, val_bleu, 0)

    if val_meteor > best_meteor:
        best_meteor = val_meteor
        best_checkpoint = checkpoint_path

print(f"\n🏆 Best checkpoint: {best_checkpoint} with METEOR = {best_meteor:.4f}")
