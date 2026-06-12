# src/metrics.py
from collections import Counter
from typing import List, Dict

def compute_unique_token_ratio(texts):
    ratios = []
    for text in texts:
        if text:
            words = text.split()
            if words:
                ratios.append(len(set(words)) / len(words))
    return sum(ratios) / len(ratios) if ratios else 0.0

def compute_average_length(texts):
    lengths = [len(text.split()) for text in texts if text]
    return sum(lengths) / len(lengths) if lengths else 0.0

def compute_coverage(predictions, references):
    total_correct = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        if pred and ref:
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            total_correct += len(pred_words.intersection(ref_words))
            total_words += len(ref_words)
    return total_correct / total_words if total_words > 0 else 0.0

def calculate_ngram_precision(predictions, references, n=4):
    precisions = {}
    for i in range(1, n+1):
        total_matches = 0
        total_predicted = 0
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                continue
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            if len(pred_tokens) < i:
                continue
            pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens)-i+1)]
            ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)]
            if not pred_ngrams:
                continue
            ref_counter = Counter(ref_ngrams)
            pred_counter = Counter(pred_ngrams)
            matches = sum(min(ref_counter[ng], pred_counter[ng]) for ng in pred_counter)
            total_matches += matches
            total_predicted += len(pred_ngrams)
        precisions[i] = total_matches / total_predicted if total_predicted > 0 else 0.0
    return precisions