# src/attention.py
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention_weights = torch.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights