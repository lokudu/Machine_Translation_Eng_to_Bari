# src/data_preprocessor.py
"""
Data preprocessing module for English-Bari parallel corpus
Properly handles Bari special characters: ŋ, ö, ü, ɛ, ɔ
"""

import re
import unicodedata
import pandas as pd
from typing import Tuple, Dict, List
from .config import DATA_CONFIG


class DataPreprocessor:
    """
    Handles cleaning and preprocessing of parallel text data
    Preserves Bari special characters: ŋ, ö, ü, ɛ, ɔ
    """
    
    # Define Bari special characters for preservation
    BARI_SPECIAL_CHARS = {
        'eng': 'ŋ',      # eng character (like n with tail)
        'o_diaeresis': 'ö',
        'u_diaeresis': 'ü',
        'open_e': 'ɛ',
        'open_o': 'ɔ'
    }
    
    # Set of all Bari special characters for quick checking
    BARI_CHARS_SET = set('ŋöüɛɔ')
    
    def __init__(self):
        self.cleaning_stats = {}
        self.bari_char_stats = {}
    
    @staticmethod
    def normalize_bari_text(text: str) -> str:
        """
        Normalize Bari text, preserving special characters
        
        Args:
            text: Raw Bari text
        
        Returns:
            Normalized text with preserved special characters
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Ensure consistent encoding of Bari special characters
        # Replace common misrepresentations
        char_mappings = {
            'Ŋ': 'ŋ',  # Uppercase eng to lowercase
            'Ö': 'ö',  # Uppercase o-diaeresis to lowercase
            'Ü': 'ü',  # Uppercase u-diaeresis to lowercase
            'Ɛ': 'ɛ',  # Uppercase open e to lowercase
            'Ɔ': 'ɔ',  # Uppercase open o to lowercase
            # Handle potential ASCII replacements
            'ng': 'ŋ',  # Common misrepresentation
            'oe': 'ö',
            'ue': 'ü',
            'e?': 'ɛ',
            'o?': 'ɔ',
        }
        
        for wrong, correct in char_mappings.items():
            if wrong in text:
                text = text.replace(wrong, correct)
        
        # Normalize Unicode (NFKC form)
        text = unicodedata.normalize('NFKC', text)
        
        return text.strip()
    
    @staticmethod
    def clean_english_text(text: str) -> str:
        """
        Clean English text
        
        Args:
            text: Raw English text
        
        Returns:
            Cleaned English text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove verse markers (e.g., "1:1", "Matthew1:1")
        text = re.sub(r'^\d+:\d+\s*', '', text)
        text = re.sub(r'^[A-Za-z]+\d+:\d+\s*', '', text)
        
        # Remove bracketed content (manuscript notes)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    def analyze_bari_characters(self, df: pd.DataFrame) -> Dict:
        """
        Analyze Bari character frequency in the dataset
        
        Args:
            df: DataFrame with 'bari_clean' column
        
        Returns:
            Dictionary with character statistics
        """
        all_text = ' '.join(df['bari_clean'].tolist())
        
        char_stats = {}
        for char in self.BARI_CHARS_SET:
            count = all_text.count(char)
            char_stats[char] = count
        
        # Also count standard characters for comparison
        total_chars = len(all_text)
        special_count = sum(char_stats.values())
        
        return {
            'total_chars': total_chars,
            'special_chars_count': special_count,
            'special_percentage': (special_count / total_chars * 100) if total_chars > 0 else 0,
            'char_counts': char_stats
        }
    
    def validate_bari_text(self, text: str) -> bool:
        """
        Validate that Bari text contains valid characters
        
        Args:
            text: Bari text to validate
        
        Returns:
            True if text contains only valid characters
        """
        if not text:
            return False
        
        # Define valid Bari characters (letters, spaces, punctuation)
        valid_chars = set('abcdefghijklmnopqrstuvwxyzŋöüɛɔ ABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:?!\'"()-')
        
        # Check if all characters are valid
        for char in text:
            if char not in valid_chars:
                return False
        
        return True
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning to entire dataset
        
        Args:
            df: DataFrame with 'Eng_Text' and 'Bari_Text' columns
        
        Returns:
            Cleaned DataFrame with 'eng_clean' and 'bari_clean' columns
        """
        original_len = len(df)
        print(f"\nCleaning dataset with {original_len:,} rows...")
        
        # Apply cleaning to English
        df['eng_clean'] = df['Eng_Text'].apply(self.clean_english_text)
        
        # Apply cleaning and normalization to Bari
        df['bari_clean'] = df['Bari_Text'].apply(self.normalize_bari_text)
        
        # Remove empty rows
        before_empty = len(df)
        df = df[(df['eng_clean'].str.len() > 0) & (df['bari_clean'].str.len() > 0)]
        self.cleaning_stats['empty_removed'] = before_empty - len(df)
        
        # Remove rows with invalid characters
        before_invalid = len(df)
        df = df[df['bari_clean'].apply(self.validate_bari_text)]
        self.cleaning_stats['invalid_chars_removed'] = before_invalid - len(df)
        
        self.cleaning_stats['after_cleaning'] = len(df)
        
        # Analyze Bari characters
        self.bari_char_stats = self.analyze_bari_characters(df)
        
        print(f"✓ Dataset cleaning complete")
        print(f"  Kept: {len(df):,} pairs ({len(df)/original_len*100:.1f}%)")
        print(f"  Removed empty: {self.cleaning_stats['empty_removed']:,}")
        print(f"  Removed invalid: {self.cleaning_stats['invalid_chars_removed']:,}")
        
        return df
    
    def filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter sentences by length constraints
        
        Args:
            df: DataFrame with cleaned data
        
        Returns:
            Filtered DataFrame
        """
        min_len = DATA_CONFIG.min_sentence_length
        max_len = DATA_CONFIG.max_sentence_length
        
        en_len = df['eng_clean'].str.split().str.len()
        ba_len = df['bari_clean'].str.split().str.len()
        
        before = len(df)
        
        # Apply length filters
        df = df[(en_len >= min_len) & (ba_len >= min_len)]
        df = df[(en_len <= max_len) & (ba_len <= max_len)]
        
        self.cleaning_stats['length_filter_removed'] = before - len(df)
        self.cleaning_stats['after_length_filter'] = len(df)
        
        print(f"✓ Length filter applied")
        print(f"  Kept: {len(df):,} pairs")
        print(f"  Removed: {self.cleaning_stats['length_filter_removed']:,} (too short/long)")
        
        return df
    
    def filter_by_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter sentences with extreme length ratio between languages
        
        Args:
            df: DataFrame with cleaned data
        
        Returns:
            Filtered DataFrame
        """
        max_ratio = DATA_CONFIG.max_length_ratio
        
        en_len = df['eng_clean'].str.split().str.len()
        ba_len = df['bari_clean'].str.split().str.len()
        ratio = en_len / ba_len
        
        before = len(df)
        df = df[(ratio < max_ratio) & (ratio > 1/max_ratio)]
        
        self.cleaning_stats['ratio_filter_removed'] = before - len(df)
        self.cleaning_stats['after_ratio_filter'] = len(df)
        
        print(f"✓ Length ratio filter applied")
        print(f"  Kept: {len(df):,} pairs")
        print(f"  Removed: {self.cleaning_stats['ratio_filter_removed']:,} (extreme length ratios)")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate sentence pairs
        
        Args:
            df: DataFrame with cleaned data
        
        Returns:
            DataFrame without duplicates
        """
        before = len(df)
        
        # Check for exact duplicates
        df = df.drop_duplicates(subset=['eng_clean', 'bari_clean'])
        
        # Also check for near-duplicates (same English, similar Bari)
        # This is a simple approach - can be extended
        
        self.cleaning_stats['duplicates_removed'] = before - len(df)
        self.cleaning_stats['after_deduplication'] = len(df)
        
        if self.cleaning_stats['duplicates_removed'] > 0:
            print(f"✓ Duplicate removal applied")
            print(f"  Kept: {len(df):,} pairs")
            print(f"  Removed: {self.cleaning_stats['duplicates_removed']:,} duplicates")
        
        return df
    
    def print_bari_character_report(self):
        """Print detailed report about Bari characters in the dataset"""
        print("\n" + "="*50)
        print("BARI CHARACTER ANALYSIS")
        print("="*50)
        
        if not self.bari_char_stats:
            print("  No character statistics available")
            return
        
        print(f"\nTotal characters in dataset: {self.bari_char_stats['total_chars']:,}")
        print(f"Special Bari characters: {self.bari_char_stats['special_chars_count']:,} ({self.bari_char_stats['special_percentage']:.2f}%)")
        
        print("\nCharacter frequencies:")
        for char, count in sorted(self.bari_char_stats['char_counts'].items(), key=lambda x: -x[1]):
            char_name = {
                'ŋ': 'eng (ŋ)',
                'ö': 'o-diaeresis (ö)',
                'ü': 'u-diaeresis (ü)',
                'ɛ': 'open e (ɛ)',
                'ɔ': 'open o (ɔ)'
            }.get(char, char)
            print(f"  {char_name}: {count:,} occurrences")
    
    def print_cleaning_report(self):
        """Print comprehensive cleaning report"""
        print("\n" + "="*50)
        print("DATA CLEANING REPORT")
        print("="*50)
        
        print("\nFiltering Statistics:")
        for key, value in self.cleaning_stats.items():
            if 'removed' in key:
                print(f"  {key}: {value:,}")
        
        print("\nFinal Counts:")
        if 'after_cleaning' in self.cleaning_stats:
            print(f"  After cleaning: {self.cleaning_stats['after_cleaning']:,}")
        if 'after_length_filter' in self.cleaning_stats:
            print(f"  After length filter: {self.cleaning_stats['after_length_filter']:,}")
        if 'after_ratio_filter' in self.cleaning_stats:
            print(f"  After ratio filter: {self.cleaning_stats['after_ratio_filter']:,}")
        if 'after_deduplication' in self.cleaning_stats:
            print(f"  Final (after deduplication): {self.cleaning_stats['after_deduplication']:,}")
        
        # Print Bari character report
        self.print_bari_character_report()
    
    def get_sample_with_characters(self, df: pd.DataFrame, num_samples: int = 5) -> List[Tuple[str, str]]:
        """
        Get sample sentences that contain Bari special characters
        
        Args:
            df: DataFrame with cleaned data
            num_samples: Number of samples to return
        
        Returns:
            List of (English, Bari) tuples
        """
        samples = []
        
        # Filter rows that contain Bari special characters
        mask = df['bari_clean'].str.contains('|'.join(self.BARI_CHARS_SET), na=False)
        char_df = df[mask]
        
        for _, row in char_df.head(num_samples).iterrows():
            samples.append((row['eng_clean'], row['bari_clean']))
        
        return samples