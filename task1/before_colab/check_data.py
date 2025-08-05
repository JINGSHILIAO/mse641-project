import json

# Check the first line of train.jsonl
with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    first_line = f.readline()
    data = json.loads(first_line)
    print("Keys in data:", list(data.keys()))
    print("Sample data structure:")
    for key, value in data.items():
        print(f"  {key}: {type(value).__name__}")
        if isinstance(value, list) and len(value) > 0:
            print(f"    First element type: {type(value[0]).__name__}") 