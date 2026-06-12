# src/decoder.py
import torch
import torch.nn as nn
from .attention import BahdanauAttention

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim + hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, decoder_hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(x))
        last_hidden = decoder_hidden[-1]
        context, _ = self.attention(last_hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, decoder_hidden = self.gru(rnn_input, decoder_hidden)
        output = output.squeeze(1)
        combined = torch.cat([output, context.squeeze(1)], dim=1)
        prediction = self.fc(combined)
        return prediction, decoder_hidden