#!/usr/bin/env python
# scripts/09_check_characters.py
"""
Verify that Bari special characters are properly preserved in the processed data
"""

import sys
import os

project_root = r'C:\Users\Lokudu James\english-bari-mt'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from src.config import PATH_CONFIG
from src.data_preprocessor import DataPreprocessor


def main():
    print("="*60)
    print("BARI CHARACTER VERIFICATION")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load processed data
    train_path = os.path.join(PATH_CONFIG.data_dir, "processed", "train.csv")
    
    if not os.path.exists(train_path):
        print(f"\n✗ Processed data not found at {train_path}")
        print("Please run preprocessing first.")
        return
    
    df = pd.read_csv(train_path)
    
    print(f"\nLoaded {len(df):,} training pairs")
    
    # Analyze characters
    char_stats = preprocessor.analyze_bari_characters(df)
    
    print("\n" + "="*50)
    print("CHARACTER FREQUENCY ANALYSIS")
    print("="*50)
    
    print(f"\nTotal characters: {char_stats['total_chars']:,}")
    print(f"Special Bari characters: {char_stats['special_chars_count']:,} ({char_stats['special_percentage']:.2f}%)")
    
    print("\nSpecial character frequencies:")
    for char, count in sorted(char_stats['char_counts'].items(), key=lambda x: -x[1]):
        char_name = {
            'ŋ': 'eng (ŋ)',
            'ö': 'o-diaeresis (ö)',
            'ü': 'u-diaeresis (ü)',
            'ɛ': 'open e (ɛ)',
            'ɔ': 'open o (ɔ)'
        }.get(char, char)
        print(f"  {char_name}: {count:,}")
    
    # Find and display sentences with each special character
    print("\n" + "="*50)
    print("EXAMPLE SENTENCES WITH SPECIAL CHARACTERS")
    print("="*50)
    
    for char in ['ŋ', 'ö', 'ü', 'ɛ', 'ɔ']:
        # Find sentences containing this character
        mask = df['bari_clean'].str.contains(char, na=False)
        examples = df[mask]['bari_clean'].head(3).tolist()
        
        if examples:
            print(f"\n'{char}' examples:")
            for i, ex in enumerate(examples, 1):
                # Highlight the character
                highlighted = ex.replace(char, f'[{char}]')
                print(f"  {i}. {highlighted[:100]}...")
        else:
            print(f"\n'{char}': No examples found")
    
    # Verify that no characters were lost
    print("\n" + "="*50)
    print("CHARACTER INTEGRITY CHECK")
    print("="*50)
    
    # Read raw data for comparison
    raw_path = os.path.join(PATH_CONFIG.data_dir, "raw", "Bari_English_Bible_corpus.csv")
    
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        
        # Count special characters in raw data
        raw_text = ' '.join(raw_df['Bari_Text'].dropna().astype(str).tolist())
        raw_counts = {}
        for char in preprocessor.BARI_CHARS_SET:
            raw_counts[char] = raw_text.count(char)
        
        print("\nCharacter preservation:")
        for char in preprocessor.BARI_CHARS_SET:
            raw_count = raw_counts.get(char, 0)
            proc_count = char_stats['char_counts'].get(char, 0)
            if raw_count > 0:
                preservation = (proc_count / raw_count) * 100
                status = "✓" if preservation > 90 else "⚠️"
                print(f"  {char}: {proc_count:,}/{raw_count:,} ({preservation:.1f}%) preserved {status}")
    
    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()