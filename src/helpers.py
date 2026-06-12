# src/helpers.py
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_english(text, vocab_size=30000):
    tokens = []
    for word in text.lower().split():
        token_id = (abs(hash(word)) % (vocab_size - 4)) + 4
        tokens.append(token_id)
    return tokens

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"