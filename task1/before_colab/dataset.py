import json
import logging
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_map = {"phrase": 0, "passage": 1, "multi": 2}
reverse_label_map = {v: k for k, v in label_map.items()} # for submission file generation

# Load data from jsonl file and extract postText and spoilerType
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            title_text = entry.get('targetTitle')
            label = entry.get('tags')
            label = label[0] if label else None
            data.append((title_text, label))
    return data

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Tokenization
def tokenize(text):
    return [tok.text.lower() for tok in nlp(text)]

# Build vocabulary
def build_vocab(data, min_freq=1):
    counter = Counter()
    for text, label in data:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# Encode text + labels
def encode_text(text, vocab, max_len=30):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids)

# Create a PyTorch Dataset
class ClickbaitDataset(Dataset):
    def __init__(self, data, vocab, max_len=30, is_test=False):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_tensor = encode_text(text, self.vocab, self.max_len)

        if self.is_test or label is None:
            return input_tensor
        
        label_tensor = torch.tensor(label_map[label])
        return input_tensor, label_tensor

# Create a DataLoader
def get_dataloader(filepath, vocab=None, batch_size=32, is_test=False):
    data = load_jsonl(filepath)

    if vocab is None:
        vocab = build_vocab(data)

    dataset = ClickbaitDataset(data, vocab, is_test=is_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)
    return loader, vocab

# ----------------- Testing ------------------
if __name__ == "__main__":
    train_data = load_jsonl("data/train.jsonl")
    logger.info(f"Loaded {len(train_data)} training examples.")
    logger.info(f"First example: {train_data[0]}")

    logger.info(f"Tokenized sample: {tokenize(train_data[0][0])}")
    vocab = build_vocab(train_data)

    dataset = ClickbaitDataset(train_data, vocab)
    loader = DataLoader(dataset, batch_size=4)

    for batch in loader:
        inputs, labels = batch
        logger.info(f"Batch input shape: {inputs.shape}")
        logger.info(f"Batch labels: {labels}")
        break

    # Optional: Test with fake test data
    test_data = [(text, None) for text, _, in train_data[:4]]  # simulate test
    test_dataset = ClickbaitDataset(test_data, vocab, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=2)

    for batch in test_loader:
        inputs = batch
        logger.info(f"Batch input shape (test): {inputs.shape}")
        break
