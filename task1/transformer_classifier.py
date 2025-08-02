import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    # forward method for distilbert-base-uncased
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]  # works for DistilBERT
        else:
            pooled = outputs.pooler_output  # BERT, RoBERTa
        return self.classifier(self.dropout(pooled))

    # forward method for roberta with mean pooling
    # def forward(self, input_ids, attention_mask):
    #     outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    #     if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
    #         # For BERT
    #         pooled = outputs.pooler_output
    #     else:
    #         # For RoBERTa and DistilBERT: mean pooling
    #         last_hidden = outputs.last_hidden_state
    #         pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    #     return self.classifier(self.dropout(pooled))

