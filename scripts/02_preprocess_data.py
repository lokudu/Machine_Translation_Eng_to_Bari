#!/usr/bin/env python
# scripts/02_preprocess_data.py
"""
Data preprocessing script with proper Bari character handling
"""

import sys
import os

# Add project root to Python path
project_root = r'C:\Users\Lokudu James\english-bari-mt'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.config import PATH_CONFIG


def main():
    print("="*60)
    print("STEP 2: Data Preprocessing with Bari Character Handling")
    print("="*60)
    
    # Initialize loader and preprocessor
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    
    # Load raw data
    raw_data_path = os.path.join(PATH_CONFIG.data_dir, "raw", "Bari_English_Bible_corpus.csv")
    
    if not os.path.exists(raw_data_path):
        print(f"\n✗ Data file not found at {raw_data_path}")
        print("Please place the CSV file in: data\\raw\\Bari_English_Bible_corpus.csv")
        return
    
    print(f"\nLoading data from: {raw_data_path}")
    df = loader.load_raw_data(raw_data_path)
    
    # Show sample of raw Bari text
    print("\n" + "-"*40)
    print("RAW BARI TEXT SAMPLES (BEFORE CLEANING)")
    print("-"*40)
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"  Raw Bari: {repr(df.iloc[i]['Bari_Text'][:100])}")
    
    # Clean dataset
    df = preprocessor.clean_dataset(df)
    
    # Show sample of cleaned Bari text
    print("\n" + "-"*40)
    print("CLEANED BARI TEXT SAMPLES (AFTER CLEANING)")
    print("-"*40)
    for i in range(min(3, len(df))):
        print(f"\nSample {i+1}:")
        print(f"  Cleaned Bari: {df.iloc[i]['bari_clean'][:100]}")
    
    # Show samples with special characters
    print("\n" + "-"*40)
    print("SAMPLES CONTAINING BARI SPECIAL CHARACTERS")
    print("-"*40)
    special_samples = preprocessor.get_sample_with_characters(df, num_samples=5)
    for i, (eng, bar) in enumerate(special_samples, 1):
        print(f"\nSample {i}:")
        print(f"  English: {eng[:80]}...")
        print(f"  Bari:    {bar[:80]}...")
        # Show which special characters are present
        specials_found = [c for c in preprocessor.BARI_CHARS_SET if c in bar]
        if specials_found:
            print(f"  Special chars: {specials_found}")
    
    # Apply filters
    df = preprocessor.filter_by_length(df)
    df = preprocessor.filter_by_ratio(df)
    df = preprocessor.remove_duplicates(df)
    
    # Create and save splits
    train_df, dev_df, test_df = loader.create_splits(df)
    loader.save_splits()
    
    # Print final reports
    preprocessor.print_cleaning_report()
    
    # Print split statistics
    print("\n" + "="*50)
    print("FINAL SPLIT STATISTICS")
    print("="*50)
    print(f"Training:   {len(train_df):,} pairs")
    print(f"Development: {len(dev_df):,} pairs")
    print(f"Test:       {len(test_df):,} pairs")
    
    # Verify Bari characters in splits
    print("\n" + "="*50)
    print("BARI CHARACTER VERIFICATION IN SPLITS")
    print("="*50)
    
    for name, split_df in [("Training", train_df), ("Development", dev_df), ("Test", test_df)]:
        special_count = 0
        for text in split_df['bari_clean']:
            if any(c in preprocessor.BARI_CHARS_SET for c in text):
                special_count += 1
        
        percentage = (special_count / len(split_df) * 100) if len(split_df) > 0 else 0
        print(f"{name}: {special_count:,}/{len(split_df):,} sentences contain special characters ({percentage:.1f}%)")
    
    print("\n✓ Data preprocessing complete!")
    print(f"\nNext step: Run python scripts/03_train_tokenizer.py")


if __name__ == "__main__":
    main()