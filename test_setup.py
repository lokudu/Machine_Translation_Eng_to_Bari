# test_setup.py
import sys
import torch
import sentencepiece as spm
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import sklearn
import sacrebleu

print("="*50)
print("ENVIRONMENT VERIFICATION")
print("="*50)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"SentencePiece version: {spm.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Seaborn version: {sns.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"SacreBLEU version: {sacrebleu.__version__}")
print("\n✓ All packages imported successfully!")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")