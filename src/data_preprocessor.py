# src/data_preprocessor.py
import re
import pandas as pd
from .config import DATA_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.cleaning_stats = {}
    
    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'^\d+:\d+\s*', '', text)
        text = re.sub(r'^[A-Za-z]+\d+:\d+\s*', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def clean_dataset(self, df):
        original_len = len(df)
        df['eng_clean'] = df['Eng_Text'].apply(self.clean_text)
        df['bari_clean'] = df['Bari_Text'].apply(self.clean_text)
        df = df[(df['eng_clean'].str.len() > 0) & (df['bari_clean'].str.len() > 0)]
        self.cleaning_stats['after_cleaning'] = len(df)
        return df
    
    def filter_by_length(self, df):
        min_len = DATA_CONFIG.min_sentence_length
        max_len = DATA_CONFIG.max_sentence_length
        en_len = df['eng_clean'].str.split().str.len()
        ba_len = df['bari_clean'].str.split().str.len()
        before = len(df)
        df = df[(en_len >= min_len) & (ba_len >= min_len)]
        df = df[(en_len <= max_len) & (ba_len <= max_len)]
        self.cleaning_stats['after_length_filter'] = len(df)
        self.cleaning_stats['length_removed'] = before - len(df)
        return df
    
    def filter_by_ratio(self, df):
        max_ratio = DATA_CONFIG.max_length_ratio
        en_len = df['eng_clean'].str.split().str.len()
        ba_len = df['bari_clean'].str.split().str.len()
        ratio = en_len / ba_len
        before = len(df)
        df = df[(ratio < max_ratio) & (ratio > 1/max_ratio)]
        self.cleaning_stats['after_ratio_filter'] = len(df)
        self.cleaning_stats['ratio_removed'] = before - len(df)
        return df
    
    def remove_duplicates(self, df):
        before = len(df)
        df = df.drop_duplicates(subset=['eng_clean', 'bari_clean'])
        self.cleaning_stats['after_deduplication'] = len(df)
        self.cleaning_stats['duplicates_removed'] = before - len(df)
        return df
    
    def print_cleaning_report(self):
        print("\n" + "="*50)
        print("DATA CLEANING REPORT")
        print("="*50)
        for key, value in self.cleaning_stats.items():
            print(f"  {key}: {value:,}")