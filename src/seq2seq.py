# src/seq2seq.py
import torch
import torch.nn as nn
import random
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim, tgt_vocab, num_layers, dropout)
    
    def forward(self, src, tgt_in, src_lens, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt_in.size(1)
        encoder_outputs, decoder_hidden = self.encoder(src, src_lens)
        mask = (src != 0).float()
        outputs = []
        decoder_input = tgt_in[:, 0].unsqueeze(1)
        for t in range(tgt_len):
            prediction, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            outputs.append(prediction)
            if teacher_forcing_ratio > 0 and random.random() < teacher_forcing_ratio:
                decoder_input = tgt_in[:, t].unsqueeze(1)
            else:
                decoder_input = prediction.argmax(1).unsqueeze(1)
        return torch.stack(outputs, dim=1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)