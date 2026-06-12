# src/bpe_tokenizer.py
"""
BPE Tokenization module using SentencePiece for Bari language
Handles training, loading, encoding, decoding, and saving of tokenizer
"""

import os
import shutil
import sentencepiece as spm
from typing import List, Optional
from .config import TOKENIZER_CONFIG, PATH_CONFIG


class BPETokenizer:
    """
    BPE Tokenizer wrapper for Bari language using SentencePiece
    """
    
    def __init__(self, vocab_size: Optional[int] = None):
        """
        Initialize the tokenizer
        
        Args:
            vocab_size: Vocabulary size (uses config if None)
        """
        self.vocab_size = vocab_size or TOKENIZER_CONFIG.vocab_size
        self.model = None
        self.model_path = None
    
    def train(self, texts: List[str], model_prefix: str = 'bpe_8k') -> None:
        """
        Train BPE tokenizer on Bari texts
        
        Args:
            texts: List of Bari sentences
            model_prefix: Prefix for output model files
        """
        print(f"\nTraining BPE tokenizer with vocab size {self.vocab_size}...")
        print(f"Number of sentences: {len(texts):,}")
        
        # Create temp directory for training
        temp_dir = "temp_tokenizer"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save texts to temporary file
        temp_file = os.path.join(temp_dir, f"{model_prefix}_train.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts[:100000]:  # Use subset for speed
                if text and len(text.strip()) > 0:
                    f.write(text.strip() + '\n')
        
        print(f"  Saved {len(texts[:100000]):,} sentences to temp file")
        
        # Train tokenizer in temp directory
        model_temp_path = os.path.join(temp_dir, model_prefix)
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_temp_path,
            vocab_size=self.vocab_size,
            character_coverage=TOKENIZER_CONFIG.character_coverage,
            model_type=TOKENIZER_CONFIG.model_type,
            pad_id=TOKENIZER_CONFIG.pad_id,
            unk_id=TOKENIZER_CONFIG.unk_id,
            bos_id=TOKENIZER_CONFIG.bos_id,
            eos_id=TOKENIZER_CONFIG.eos_id,
            pad_piece=TOKENIZER_CONFIG.pad_piece,
            unk_piece=TOKENIZER_CONFIG.unk_piece,
            bos_piece=TOKENIZER_CONFIG.bos_piece,
            eos_piece=TOKENIZER_CONFIG.eos_piece,
            num_threads=4,
            input_sentence_size=1000000
        )
        
        # Load model from temp directory
        self.model = spm.SentencePieceProcessor()
        temp_model_file = f"{model_temp_path}.model"
        self.model.load(temp_model_file)
        
        # Create models directory
        os.makedirs(PATH_CONFIG.models_dir, exist_ok=True)
        
        # Save model to permanent location
        self.model_path = os.path.join(PATH_CONFIG.models_dir, f"{model_prefix}.model")
        vocab_path = os.path.join(PATH_CONFIG.models_dir, f"{model_prefix}.vocab")
        
        # Copy files from temp to permanent location
        shutil.copy(temp_model_file, self.model_path)
        shutil.copy(f"{model_temp_path}.vocab", vocab_path)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"✓ Trained BPE tokenizer with vocab size {self.model.vocab_size()}")
        print(f"  Model saved to: {self.model_path}")
    
    def load(self, model_path: str) -> None:
        """
        Load pre-trained tokenizer from file
        
        Args:
            model_path: Path to the .model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found at {model_path}")
        
        self.model = spm.SentencePieceProcessor()
        self.model.load(model_path)
        self.model_path = model_path
        self.vocab_size = self.model.vocab_size()
        print(f"✓ Loaded tokenizer from {model_path}")
        print(f"  Vocabulary size: {self.vocab_size:,}")
    
    def save(self, path: str) -> None:
        """
        Save tokenizer model to disk
        
        Args:
            path: Destination path for the .model file
        """
        if self.model is None:
            raise ValueError("No tokenizer loaded. Train or load first.")
        
        # If source and destination are the same, skip copying
        if self.model_path and os.path.abspath(self.model_path) == os.path.abspath(path):
            print(f"✓ Tokenizer already at {path}")
            return
        
        # Create destination directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Copy the model file
        if self.model_path and os.path.exists(self.model_path):
            shutil.copy(self.model_path, path)
            print(f"✓ Saved tokenizer to {path}")
        else:
            # If no file path, save the loaded model
            self.model.save(path)
            print(f"✓ Saved tokenizer to {path}")
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded. Call train() or load() first.")
        return self.model.encode_as_ids(text)
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded. Call train() or load() first.")
        return self.model.decode_ids(ids)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text to subword pieces
        
        Args:
            text: Input text string
        
        Returns:
            List of subword pieces
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded.")
        return self.model.encode_as_pieces(text)
    
    def get_vocab_size(self) -> int:
        """
        Return vocabulary size
        
        Returns:
            Vocabulary size
        """
        return self.model.vocab_size() if self.model else 0
    
    def piece_to_id(self, piece: str) -> int:
        """
        Convert a piece to its token ID
        
        Args:
            piece: Subword piece string
        
        Returns:
            Token ID
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded.")
        return self.model.piece_to_id(piece)
    
    def id_to_piece(self, idx: int) -> str:
        """
        Convert a token ID to its piece
        
        Args:
            idx: Token ID
        
        Returns:
            Subword piece string
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded.")
        return self.model.id_to_piece(idx)
    
    def compute_oov_rate(self, texts: List[str], sample_size: int = 5000) -> float:
        """
        Compute Out-of-Vocabulary rate on sample texts
        
        Args:
            texts: List of texts to evaluate
            sample_size: Number of sentences to sample
        
        Returns:
            OOV rate as a float between 0 and 1
        """
        if self.model is None:
            raise ValueError("Tokenizer not loaded.")
        
        oov_count = 0
        total_words = 0
        
        for text in texts[:sample_size]:
            words = text.split()
            total_words += len(words)
            for word in words:
                pieces = self.model.encode_as_pieces(word)
                # If word is represented as a single piece that's not the word itself
                if len(pieces) == 1 and pieces[0] == word:
                    if self.model.piece_to_id(word) == TOKENIZER_CONFIG.unk_id:
                        oov_count += 1
        
        oov_rate = oov_count / total_words if total_words > 0 else 1.0
        print(f"  OOV rate: {oov_rate:.2%} (based on {sample_size} sentences)")
        return oov_rate
    
    def test_tokenizer(self, sample_text: str) -> None:
        """
        Test the tokenizer on a sample text
        
        Args:
            sample_text: Sample Bari text to test
        """
        print("\n" + "="*50)
        print("TOKENIZER TEST")
        print("="*50)
        print(f"Original: {sample_text[:100]}...")
        
        encoded = self.encode(sample_text)
        print(f"Encoded (first 20): {encoded[:20]}")
        
        decoded = self.decode(encoded)
        print(f"Decoded: {decoded[:100]}...")
        
        pieces = self.encode_as_pieces(sample_text)
        print(f"Pieces (first 15): {pieces[:15]}")
        
        print(f"Vocabulary size: {self.get_vocab_size():,}")
    
    def get_info(self) -> dict:
        """
        Get information about the tokenizer
        
        Returns:
            Dictionary with tokenizer information
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "vocab_size": self.get_vocab_size(),
            "model_path": self.model_path,
            "pad_id": TOKENIZER_CONFIG.pad_id,
            "unk_id": TOKENIZER_CONFIG.unk_id,
            "bos_id": TOKENIZER_CONFIG.bos_id,
            "eos_id": TOKENIZER_CONFIG.eos_id
        }