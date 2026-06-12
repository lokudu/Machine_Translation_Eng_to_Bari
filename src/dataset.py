# src/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
from .helpers import tokenize_english

class BariTranslationDataset(Dataset):
    def __init__(self, df, bpe_tokenizer, max_len=50):
        self.df = df.reset_index(drop=True)
        self.tokenizer = bpe_tokenizer
        self.max_len = max_len
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_ids = tokenize_english(row['eng_clean'])[:self.max_len]
        tgt_ids = self.tokenizer.encode(row['bari_clean'])[:self.max_len - 2]
        tgt_ids = [self.bos_id] + tgt_ids + [self.eos_id]
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt_in': torch.tensor(tgt_ids[:-1], dtype=torch.long),
            'tgt_out': torch.tensor(tgt_ids[1:], dtype=torch.long)
        }

def collate_batch(batch):
    src = [item['src'] for item in batch]
    tgt_in = [item['tgt_in'] for item in batch]
    tgt_out = [item['tgt_out'] for item in batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_in_padded = torch.nn.utils.rnn.pad_sequence(tgt_in, batch_first=True, padding_value=0)
    tgt_out_padded = torch.nn.utils.rnn.pad_sequence(tgt_out, batch_first=True, padding_value=0)
    src_lens = torch.tensor([len(s) for s in src])
    return {'src': src_padded, 'tgt_in': tgt_in_padded, 'tgt_out': tgt_out_padded, 'src_lens': src_lens}