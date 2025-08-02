from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
import json

label_map = {"phrase": 0, "passage": 1, "multi": 2}

class ClickbaitSpoilerTypeDataset(Dataset):
    def __init__(self, filepath, tokenizer_name='bert-base-uncased', max_len=64, is_test=False):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.is_test = is_test
        self.data = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                post_text = entry['postText']
                target_title = entry['targetTitle'] 
                # concatenate postText and targetTitle
                combined_text = f"{post_text} {self.tokenizer.sep_token} {target_title}"

                label = entry.get('tags')
                label = label[0] if label else None
                self.data.append((combined_text, label))

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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

        if not self.is_test and label is not None:
            item['labels'] = torch.tensor(label_map[label])
        return item

def get_dataloader(filepath, batch_size=32, tokenizer_name='bert-base-uncased', max_len=64, is_test=False):
    dataset = ClickbaitSpoilerTypeDataset(filepath, tokenizer_name, max_len, is_test)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not is_test)