#!/usr/bin/env python
# scripts/03_train_tokenizer.py
"""
Train BPE tokenizer for Bari language using SentencePiece
Creates tokenizer model and vocabulary files
"""

import sys
import os

# Add project root to Python path
project_root = r'C:\Users\Lokudu James\english-bari-mt'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.bpe_tokenizer import BPETokenizer
from src.config import PATH_CONFIG


def main():
    print("="*60)
    print("STEP 3: Training BPE Tokenizer")
    print("="*60)
    
    # Check if processed data exists
    train_path = os.path.join(PATH_CONFIG.data_dir, "processed", "train.csv")
    
    if not os.path.exists(train_path):
        print(f"\n✗ Train file not found at {train_path}")
        print("Please run scripts/02_preprocess_data.py first.")
        return
    
    # Load training data
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    bari_texts = train_df['bari_clean'].tolist()
    print(f"✓ Loaded {len(bari_texts):,} sentences for tokenizer training")
    
    # Show sample
    if bari_texts:
        print(f"\nSample text: {bari_texts[0][:100]}...")
    
    # Train tokenizer with vocab size 8000
    print("\n" + "-"*40)
    print("Training BPE Tokenizer (vocab_size=8000)")
    print("-"*40)
    
    tokenizer = BPETokenizer(vocab_size=8000)
    tokenizer.train(bari_texts, model_prefix='bpe_8k')
    
    # The train() method already saves the model to models directory
    # No need to call save() again
    
    # Test the tokenizer
    print("\n" + "-"*40)
    print("Testing Tokenizer")
    print("-"*40)
    
    if bari_texts:
        sample_text = bari_texts[0]
        tokenizer.test_tokenizer(sample_text)
    
    # Compute OOV rate on a small sample
    print("\n" + "-"*40)
    print("Computing OOV Rate")
    print("-"*40)
    oov_rate = tokenizer.compute_oov_rate(bari_texts[:1000], sample_size=1000)
    
    # Print tokenizer info
    print("\n" + "-"*40)
    print("Tokenizer Information")
    print("-"*40)
    info = tokenizer.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Verify files were created
    print("\n" + "-"*40)
    print("Verifying Output Files")
    print("-"*40)
    
    model_file = os.path.join(PATH_CONFIG.models_dir, "bpe_8k.model")
    vocab_file = os.path.join(PATH_CONFIG.models_dir, "bpe_8k.vocab")
    
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file)
        print(f"✓ Model file: {model_file} ({file_size:,} bytes)")
    else:
        print(f"✗ Model file not found: {model_file}")
    
    if os.path.exists(vocab_file):
        file_size = os.path.getsize(vocab_file)
        print(f"✓ Vocab file: {vocab_file} ({file_size:,} bytes)")
    else:
        print(f"✗ Vocab file not found: {vocab_file}")
    
    print("\n" + "="*60)
    print("✓ Tokenizer training complete!")
    print("="*60)
    print(f"\nFiles saved to: {PATH_CONFIG.models_dir}")
    print("  - bpe_8k.model (tokenizer model)")
    print("  - bpe_8k.vocab (vocabulary file)")
    print("\nNext step: Run python scripts/04_train_model.py")


if __name__ == "__main__":
    main()