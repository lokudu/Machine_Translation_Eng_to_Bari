from setuptools import setup, find_packages

setup(
    name="english-bari-mt",
    version="1.0.0",
    description="English to Bari Neural Machine Translation",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "sentencepiece>=0.1.99",
        "sacrebleu>=2.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
)