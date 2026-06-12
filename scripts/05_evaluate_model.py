#!/usr/bin/env python
# scripts/05_evaluate_model.py
import sys
import pandas as pd
import torch
sys.path.append('..')
from src.config import PATH_CONFIG
from src.dataset import BariTranslationDataset
from src.seq2seq import Seq2Seq
from src.bpe_tokenizer import BPETokenizer
from src.evaluator import ModelEvaluator
from src.helpers import get_device

def main():
    print("="*60)
    print("STEP 5: Evaluating Model")
    print("="*60)
    device = get_device()
    test_df = pd.read_csv(f"{PATH_CONFIG.data_dir}processed/test.csv")
    tokenizer = BPETokenizer()
    tokenizer.load(f"{PATH_CONFIG.models_dir}bpe_8k.model")
    model = Seq2Seq(
        src_vocab=30000,
        tgt_vocab=tokenizer.get_vocab_size(),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    )
    model.load_state_dict(torch.load(f"{PATH_CONFIG.models_dir}best_model.pt", map_location=device))
    model = model.to(device)
    test_dataset = BariTranslationDataset(test_df, tokenizer, max_len=50)
    evaluator = ModelEvaluator(model, tokenizer, device)
    print("\nEvaluating with Beam-5...")
    bleu, chrf, _, _ = evaluator.evaluate(test_dataset, beam_width=5)
    print(f"\nTest Set Results:")
    print(f"  BLEU Score: {bleu:.2f}")
    print(f"  chrF Score: {chrf:.2f}")

if __name__ == "__main__":
    main()