import json
import logging
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_map = {"phrase": 0, "passage": 1, "multi": 2}
reverse_label_map = {v: k for k, v in label_map.items()}  # For submission file generation

# Load spaCy model for tokenization
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Tokenization
def tokenize(text):
    return [tok.text.lower() for tok in nlp(text)]

# Load data from jsonl file and extract combined input + label
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            title = entry.get('targetTitle', "")
            paragraphs = " ".join(entry.get('targetParagraphs', []))
            full_input = f"{title} [SEP] {paragraphs}"

            label = entry.get('tags')
            label = label[0] if label else None
            data.append((full_input, label))
    return data

# Build vocabulary
def build_vocab(data, min_freq=1):
    counter = Counter()
    for text, _ in data:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# Encode text as tensor
# max_len of 200 does not capture all paragraphs but it's more balanced
def encode_text(text, vocab, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))
    return torch.tensor(ids)

# PyTorch Dataset
class ClickbaitDataset(Dataset):
    def __init__(self, data, vocab, max_len=200, is_test=False):
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

# DataLoader helper
def get_dataloader(filepath, vocab=None, batch_size=32, max_len=200, is_test=False):
    data = load_jsonl(filepath)

    if vocab is None:
        vocab = build_vocab(data)

    dataset = ClickbaitDataset(data, vocab, max_len=max_len, is_test=is_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)
    return loader, vocab
