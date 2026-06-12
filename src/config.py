# src/config.py
"""
Configuration file for English to Bari Translation Project
All hyperparameters and paths are defined here for easy modification
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """Data-related configuration"""
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    min_sentence_length: int = 3
    max_sentence_length: int = 100
    max_length_ratio: float = 3.0


@dataclass
class TokenizerConfig:
    """Tokenizer configuration for SentencePiece BPE"""
    vocab_size: int = 8000
    character_coverage: float = 0.9995
    model_type: str = 'bpe'
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    pad_piece: str = '<pad>'
    unk_piece: str = '<unk>'
    bos_piece: str = '<s>'
    eos_piece: str = '</s>'


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    src_vocab_size: int = 30000
    tgt_vocab_size: int = 8000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    max_seq_len: int = 50


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    num_epochs: int = 10
    teacher_forcing_ratio: float = 0.5
    teacher_forcing_decay: float = 0.03
    early_stopping_patience: int = 5


@dataclass
class DecodingConfig:
    """Decoding configuration for inference"""
    beam_widths: Tuple[int, ...] = (1, 3, 5)
    repetition_penalty: float = 1.2
    length_penalty_alpha: float = 0.8
    max_decoding_len: int = 50


@dataclass
class PathConfig:
    """File path configuration"""
    base_dir: str = '.'
    data_dir: str = 'data/'
    models_dir: str = 'models/'
    figures_dir: str = 'figures/'
    results_dir: str = 'results/'
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        # Create main directories
        for dir_path in [self.data_dir, self.models_dir, self.figures_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        
        print(f"✓ Directory structure ready at: {os.path.abspath(self.base_dir)}")


# Create singleton instances for easy import
DATA_CONFIG = DataConfig()
TOKENIZER_CONFIG = TokenizerConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
DECODING_CONFIG = DecodingConfig()
PATH_CONFIG = PathConfig()


def print_config():
    """Print all configuration settings for verification"""
    print("=" * 60)
    print("CONFIGURATION SETTINGS")
    print("=" * 60)
    
    print("\n[Data Configuration]")
    for key, value in DATA_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Tokenizer Configuration]")
    for key, value in TOKENIZER_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Model Configuration]")
    for key, value in MODEL_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Training Configuration]")
    for key, value in TRAINING_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Decoding Configuration]")
    for key, value in DECODING_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n[Path Configuration]")
    for key, value in PATH_CONFIG.__dict__.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


# Optional: Verify paths when module is loaded
if __name__ == "__main__":
    print_config()