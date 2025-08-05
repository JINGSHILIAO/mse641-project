import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1,
                 bidirectional=False, dropout=0.5, num_classes=3, padding_idx=0):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, x):
        x = self.embedding(x)  
        output, _ = self.lstm(x) 

        # Global max pooling over time
        x = torch.max(output, dim=1).values  
        x = self.dropout(x)
        logits = self.fc(x)
        return logits