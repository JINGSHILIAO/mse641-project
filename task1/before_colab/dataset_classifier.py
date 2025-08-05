# dataset_classifier.py
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel

class SpoilerDataset(Dataset):
    def __init__(self, data, model_name, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model_name = model_name
        self.uses_token_type_ids = 'token_type_ids' in self.tokenizer.model_input_names

        # Filter out invalid entries during init
        self.data = []
        for i, item in enumerate(data):
            if all(k in item for k in ("targetTitle", "targetParagraphs", "tags")):
                self.data.append(item)
            else:
                print(f"Skipping invalid entry at index {i}: {item}")

        # Map spoiler type string to integer label
        self.label_map = {"phrase": 0, "passage": 1, "multi": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        title = item['targetTitle']
        paragraphs = item['targetParagraphs']
        paragraph = " ".join(paragraphs) if isinstance(paragraphs, list) else str(paragraphs)

        encoding = self.tokenizer(
            title,
            paragraph,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Convert spoiler tag to integer label
        spoiler_tag = item['tags'][0] if isinstance(item['tags'], list) and item['tags'] else None
        label = self.label_map.get(spoiler_tag, -1)
        if label == -1:
            raise ValueError(f"Invalid spoiler tag: {spoiler_tag} in item: {item}")

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': label
        }
        
        # Only include token_type_ids if the model uses them
        if self.uses_token_type_ids and 'token_type_ids' in encoding:
            result['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        return result

class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.uses_token_type_ids = 'token_type_ids' in self.bert.forward.__code__.co_varnames

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.uses_token_type_ids and token_type_ids is not None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)