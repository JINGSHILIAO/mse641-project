from transformers import AutoModelForSeq2SeqLM

checkpoint_path = "./checkpoints/distilbart/checkpoint-336"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

print("Generation config from checkpoint:")
print(model.generation_config)

model.generation_config.num_beams = 1
model.generation_config.min_length = 1
model.generation_config.length_penalty = 1.0
model.generation_config.early_stopping = False

print("\nAfter overriding:")
print(model.generation_config)
