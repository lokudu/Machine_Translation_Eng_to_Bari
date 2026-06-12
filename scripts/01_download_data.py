#!/usr/bin/env python
# scripts/01_download_data.py
import sys
import os
sys.path.append('..')
from src.config import PATH_CONFIG

def main():
    print("="*60)
    print("STEP 1: Downloading Data")
    print("="*60)
    print("\nPlease download the dataset from Kaggle manually:")
    print("URL: https://www.kaggle.com/datasets/lokudu/english-to-bari-bible-machine-translation")
    print(f"\nPlace the downloaded CSV file in: {PATH_CONFIG.data_dir}raw/")
    print("Expected filename: Bari_English_Bible_corpus.csv")
    raw_data_path = f"{PATH_CONFIG.data_dir}raw/Bari_English_Bible_corpus.csv"
    if os.path.exists(raw_data_path):
        print(f"\n✓ Data file found at {raw_data_path}")
    else:
        print(f"\n✗ Data file not found. Please download manually.")

if __name__ == "__main__":
    main()