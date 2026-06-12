#!/usr/bin/env python
# scripts/02_preprocess_data.py
import sys
import os

# Add the project root directory to Python path
project_root = r'C:\Users\Lokudu James\english-bari-mt'
sys.path.insert(0, project_root)

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.config import PATH_CONFIG

def main():
    print("="*60)
    print("STEP 2: Data Preprocessing")
    print("="*60)
    
    loader = DataLoader()
    raw_data_path = f"{PATH_CONFIG.data_dir}raw/Bari_English_Bible_corpus.csv"
    
    # Check if file exists
    if not os.path.exists(raw_data_path):
        print(f"\n✗ Data file not found at {raw_data_path}")
        print("Please place the CSV file in: data\\raw\\Bari_English_Bible_corpus.csv")
        return
    
    df = loader.load_raw_data(raw_data_path)
    preprocessor = DataPreprocessor()
    df = preprocessor.clean_dataset(df)
    df = preprocessor.filter_by_length(df)
    df = preprocessor.filter_by_ratio(df)
    df = preprocessor.remove_duplicates(df)
    loader.create_splits(df)
    loader.save_splits()
    preprocessor.print_cleaning_report()
    print("\n✓ Data preprocessing complete!")

if __name__ == "__main__":
    main()