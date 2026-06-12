# src/data_loader.py
"""
Data loading module for English-Bari parallel corpus
Handles loading CSV files and creating train/dev/test splits
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .config import DATA_CONFIG, PATH_CONFIG


class DataLoader:
    """
    Handles loading and splitting of parallel corpus data
    """
    
    def __init__(self):
        """Initialize the DataLoader with empty data containers"""
        self.raw_data = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None
    
    def load_raw_data(self, filepath=None):
        """
        Load raw data from CSV file
        
        Args:
            filepath: Path to CSV file (uses config if None)
        
        Returns:
            DataFrame with raw data
        """
        if filepath is None:
            filepath = os.path.join(PATH_CONFIG.data_dir, "raw", "Bari_English_Bible_corpus.csv")
        
        print(f"Loading data from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found at {filepath}")
        
        self.raw_data = pd.read_csv(filepath, encoding='utf-8')
        print(f"✓ Loaded {len(self.raw_data):,} raw sentence pairs")
        print(f"  Columns: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def get_statistics(self):
        """
        Get basic statistics about the loaded data
        
        Returns:
            Dictionary with statistics
        """
        if self.raw_data is None:
            return {}
        
        stats = {
            'total_pairs': len(self.raw_data),
            'columns': list(self.raw_data.columns),
            'memory_usage_mb': self.raw_data.memory_usage(deep=True).sum() / (1024 ** 2)
        }
        return stats
    
    def create_splits(self, df):
        """
        Create train/dev/test splits
        
        Args:
            df: DataFrame with cleaned data (must have 'eng_clean' and 'bari_clean' columns)
        
        Returns:
            Tuple of (train_df, dev_df, test_df)
        """
        train_ratio = DATA_CONFIG.train_ratio
        dev_ratio = DATA_CONFIG.dev_ratio
        test_ratio = DATA_CONFIG.test_ratio
        random_state = DATA_CONFIG.random_seed
        
        print(f"\nCreating splits with ratios: Train={train_ratio}, Dev={dev_ratio}, Test={test_ratio}")
        
        # First split: separate test set
        train_dev, test = train_test_split(
            df, 
            test_size=test_ratio, 
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: separate dev from train
        # Calculate dev proportion relative to train_dev
        dev_proportion = dev_ratio / (train_ratio + dev_ratio)
        
        train, dev = train_test_split(
            train_dev, 
            test_size=dev_proportion, 
            random_state=random_state,
            shuffle=True
        )
        
        # Reset indices
        self.train_data = train.reset_index(drop=True)
        self.dev_data = dev.reset_index(drop=True)
        self.test_data = test.reset_index(drop=True)
        
        print(f"✓ Created splits:")
        print(f"  Training:   {len(self.train_data):,} pairs ({len(self.train_data)/len(df)*100:.1f}%)")
        print(f"  Development: {len(self.dev_data):,} pairs ({len(self.dev_data)/len(df)*100:.1f}%)")
        print(f"  Test:       {len(self.test_data):,} pairs ({len(self.test_data)/len(df)*100:.1f}%)")
        
        return self.train_data, self.dev_data, self.test_data
    
    def save_splits(self):
        """
        Save the split data to CSV files in the processed directory
        """
        processed_dir = os.path.join(PATH_CONFIG.data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        if self.train_data is not None:
            train_path = os.path.join(processed_dir, "train.csv")
            self.train_data.to_csv(train_path, index=False)
            print(f"✓ Saved training data to {train_path}")
        
        if self.dev_data is not None:
            dev_path = os.path.join(processed_dir, "dev.csv")
            self.dev_data.to_csv(dev_path, index=False)
            print(f"✓ Saved development data to {dev_path}")
        
        if self.test_data is not None:
            test_path = os.path.join(processed_dir, "test.csv")
            self.test_data.to_csv(test_path, index=False)
            print(f"✓ Saved test data to {test_path}")
    
    def load_splits(self):
        """
        Load previously saved splits from CSV files
        
        Returns:
            Tuple of (train_df, dev_df, test_df)
        """
        processed_dir = os.path.join(PATH_CONFIG.data_dir, "processed")
        
        train_path = os.path.join(processed_dir, "train.csv")
        dev_path = os.path.join(processed_dir, "dev.csv")
        test_path = os.path.join(processed_dir, "test.csv")
        
        if not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
            raise FileNotFoundError("Split files not found. Run create_splits() first.")
        
        self.train_data = pd.read_csv(train_path)
        self.dev_data = pd.read_csv(dev_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"✓ Loaded splits:")
        print(f"  Training: {len(self.train_data):,} pairs")
        print(f"  Development: {len(self.dev_data):,} pairs")
        print(f"  Test: {len(self.test_data):,} pairs")
        
        return self.train_data, self.dev_data, self.test_data
    
    def print_summary(self):
        """
        Print a summary of the loaded data
        """
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        if self.train_data is not None:
            print(f"\nTraining set: {len(self.train_data):,} pairs")
            print(f"  English avg length: {self.train_data['eng_clean'].str.split().str.len().mean():.1f} words")
            print(f"  Bari avg length:    {self.train_data['bari_clean'].str.split().str.len().mean():.1f} words")
        
        if self.dev_data is not None:
            print(f"\nDevelopment set: {len(self.dev_data):,} pairs")
            print(f"  English avg length: {self.dev_data['eng_clean'].str.split().str.len().mean():.1f} words")
            print(f"  Bari avg length:    {self.dev_data['bari_clean'].str.split().str.len().mean():.1f} words")
        
        if self.test_data is not None:
            print(f"\nTest set: {len(self.test_data):,} pairs")
            print(f"  English avg length: {self.test_data['eng_clean'].str.split().str.len().mean():.1f} words")
            print(f"  Bari avg length:    {self.test_data['bari_clean'].str.split().str.len().mean():.1f} words")