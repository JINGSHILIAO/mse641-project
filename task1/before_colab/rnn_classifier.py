import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        rnn_type="lstm",
        num_layers=1,
        dropout=0.3,
        padding_idx=0,
        pooling="mean"
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn_type = rnn_type.lower()
        self.pooling = pooling.lower()

        bidirectional = "bi" in self.rnn_type
        rnn_class = nn.LSTM if "lstm" in self.rnn_type else nn.GRU

        self.rnn = rnn_class(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  
        rnn_out, _ = self.rnn(embedded) 

        if self.pooling == "last":
            pooled = rnn_out[:, -1, :]  
        elif self.pooling == "max":
            pooled, _ = torch.max(rnn_out, dim=1)
        else:  # default to mean pooling
            pooled = torch.mean(rnn_out, dim=1)

        output = self.fc(self.dropout(pooled))
        return output