#!/usr/bin/env python
# scripts/04_train_model.py
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.append('..')
from src.config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG
from src.dataset import BariTranslationDataset, collate_batch
from src.seq2seq import Seq2Seq
from src.bpe_tokenizer import BPETokenizer
from src.trainer import ModelTrainer
from src.helpers import set_seed, get_device, count_parameters

def main():
    print("="*60)
    print("STEP 4: Training Seq2Seq Model")
    print("="*60)
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")
    train_df = pd.read_csv(f"{PATH_CONFIG.data_dir}processed/train.csv")
    dev_df = pd.read_csv(f"{PATH_CONFIG.data_dir}processed/dev.csv")
    tokenizer = BPETokenizer()
    tokenizer.load(f"{PATH_CONFIG.models_dir}bpe_8k.model")
    train_dataset = BariTranslationDataset(train_df, tokenizer, MODEL_CONFIG.max_seq_len)
    dev_dataset = BariTranslationDataset(dev_df, tokenizer, MODEL_CONFIG.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=TRAINING_CONFIG.batch_size, shuffle=False, collate_fn=collate_batch)
    model = Seq2Seq(
        src_vocab=MODEL_CONFIG.src_vocab_size,
        tgt_vocab=tokenizer.get_vocab_size(),
        embed_dim=MODEL_CONFIG.embedding_dim,
        hidden_dim=MODEL_CONFIG.hidden_dim,
        num_layers=MODEL_CONFIG.num_layers,
        dropout=MODEL_CONFIG.dropout
    )
    print(f"\nModel parameters: {count_parameters(model):,}")
    trainer = ModelTrainer(model, device)
    trainer.train(train_loader, dev_loader, epochs=TRAINING_CONFIG.num_epochs,
                  teacher_forcing_start=TRAINING_CONFIG.teacher_forcing_ratio,
                  teacher_forcing_end=TRAINING_CONFIG.teacher_forcing_ratio - TRAINING_CONFIG.teacher_forcing_decay * TRAINING_CONFIG.num_epochs)
    print(f"\n✓ Training complete! Best validation loss: {trainer.best_val_loss:.4f}")

if __name__ == "__main__":
    main()