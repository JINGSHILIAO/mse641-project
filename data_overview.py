import json
import os
from collections import Counter

def count_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

print("Train examples:", count_jsonl("data/train.jsonl"))
print("Val examples:", count_jsonl("data/val.jsonl"))

# Examine class distribution
TRAIN_FILE = 'data/train.jsonl'
VAL_FILE = 'data/val.jsonl'

def load_and_count_labels(filepath):
    counter = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            tags = entry.get('tags', [])
            if tags:
                tag = tags[0]
                counter[tag] += 1
    return counter

train_counts = load_and_count_labels(TRAIN_FILE)
val_counts = load_and_count_labels(VAL_FILE)

def print_counts(title, counts):
    print(f"\n{title} class distribution:")
    total = sum(counts.values())
    for cls, count in counts.items():
        percentage = (count / total) * 100
        print(f"- {cls}: {count} ({percentage:.2f}%)")
    print(f"Total: {total}")

print_counts("Training Set", train_counts)
print_counts("Validation Set", val_counts)