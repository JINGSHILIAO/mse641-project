import torch.nn as nn
import torch.nn.functional as F
import torch

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=3,
                 kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5,
                 padding_idx=0, activation='relu'):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # Activation function
        self.activation = {
            'relu': F.relu,
            'gelu': F.gelu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
        }.get(activation, F.relu)

    def forward(self, x):
        x = self.embedding(x)          
        x = x.permute(0, 2, 1)         

        conv_outputs = [
            F.max_pool1d(self.activation(conv(x)), kernel_size=conv(x).shape[2]).squeeze(2)
            for conv in self.convs
        ]

        x = torch.cat(conv_outputs, dim=1) 
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
