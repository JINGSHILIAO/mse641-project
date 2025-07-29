from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json

label_map = {"phrase": 0, "passage": 1, "multi": 2}

class BertClickbaitDataset(Dataset):
    def __init__(self, filepath, tokenizer_name='bert-base-uncased', max_len=64, is_test=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.is_test = is_test
        self.data = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                title = entry['targetTitle'] #use only title for simple baseline experiment
                label = entry.get('tags')
                label = label[0] if label else None
                self.data.append((title, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        if not self.is_test and label is not None:
            item['labels'] = torch.tensor(label_map[label])
        return item

def get_dataloader(filepath, batch_size=32, tokenizer_name='bert-base-uncased', max_len=64, is_test=False):
    dataset = BertClickbaitDataset(filepath, tokenizer_name, max_len, is_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)