
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        # Activation function selection
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        x = self.embedding(x)  
        x = x.permute(0, 2, 1)  

        conv_outputs = []
        for conv in self.convs:
            conv_out = self.activation(conv(x))  
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2])  
            conv_outputs.append(pooled.squeeze(2)) 

        x = torch.cat(conv_outputs, dim=1) 
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
