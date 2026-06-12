# src/visualization.py
"""
Visualization utilities for model training, evaluation, and results
Includes: loss curves, BLEU comparison, error distribution, length distribution,
attention maps, training progress, and result tables
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Set style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def set_plot_style():
    """Set consistent style for all plots"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                         save_path: str = 'figures/training_curves.png'):
    """
    Plot training and validation loss curves with annotations
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    epochs = range(1, len(train_losses) + 1)
    
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = val_losses.index(min(val_losses)) + 1
    axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, linewidth=2)
    axes[0].text(best_epoch + 0.5, max(train_losses), f'Best Model\nEpoch {best_epoch}', 
                 fontsize=10, ha='left', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Loss difference (overfitting indicator)
    loss_diff = [t - v for t, v in zip(train_losses, val_losses)]
    colors = ['green' if d < 0 else 'red' for d in loss_diff]
    axes[1].bar(epochs, loss_diff, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss - Val Loss')
    axes[1].set_title('Overfitting Indicator (Negative = Good)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    if any(d > 0 for d in loss_diff):
        axes[1].text(0.02, 0.95, '⚠️ Overfitting detected\nwhen difference becomes positive', 
                     transform=axes[1].transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {save_path}")


def plot_bleu_comparison(results: Dict, save_path: str = 'figures/bleu_comparison.png'):
    """
    Plot BLEU score comparison across decoding strategies
    
    Args:
        results: Dictionary with results for different decoding strategies
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    strategies = list(results.keys())
    bleu_scores = [results[s]['bleu'] for s in strategies]
    chrf_scores = [results[s]['chrf'] for s in strategies]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Plot 1: BLEU scores
    bars1 = axes[0].bar(strategies, bleu_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Decoding Strategy')
    axes[0].set_ylabel('BLEU Score')
    axes[0].set_title('BLEU Scores by Decoding Strategy')
    axes[0].set_ylim(0, max(bleu_scores) + 0.5)
    
    for bar, score in zip(bars1, bleu_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: chrF scores
    bars2 = axes[1].bar(strategies, chrf_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Decoding Strategy')
    axes[1].set_ylabel('chrF Score')
    axes[1].set_title('chrF Scores by Decoding Strategy')
    axes[1].set_ylim(0, max(chrf_scores) + 2)
    
    for bar, score in zip(bars2, chrf_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved BLEU comparison to {save_path}")


def plot_error_distribution(error_data: Dict, save_path: str = 'figures/error_distribution.png'):
    """
    Plot error distribution bar chart and pie chart
    
    Args:
        error_data: Dictionary with error types as keys and counts as values
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    error_types = list(error_data.keys())
    counts = list(error_data.values())
    total = sum(counts)
    percentages = [c/total * 100 for c in counts]
    
    colors = plt.cm.Set3(range(len(error_types)))
    
    # Plot 1: Horizontal bar chart
    bars = axes[0].barh(error_types, counts, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_xlabel('Count (out of 100)')
    axes[0].set_ylabel('Error Type')
    axes[0].set_title('Error Distribution by Type')
    axes[0].invert_yaxis()
    
    for bar, count, pct in zip(bars, counts, percentages):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{count} ({pct:.1f}%)', ha='left', va='center', fontsize=9)
    
    # Plot 2: Pie chart
    wedges, texts, autotexts = axes[1].pie(counts, labels=error_types, autopct='%1.1f%%',
                                            colors=colors, startangle=90,
                                            textprops={'fontsize': 9})
    axes[1].set_title('Error Distribution (Percentage)')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved error distribution to {save_path}")


def plot_length_distribution(train_df, dev_df, test_df, save_path: str = 'figures/length_distribution.png'):
    """
    Plot sentence length distribution across splits
    
    Args:
        train_df: Training DataFrame
        dev_df: Development DataFrame
        test_df: Test DataFrame
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # English lengths
    train_en_len = train_df['eng_clean'].str.split().str.len()
    dev_en_len = dev_df['eng_clean'].str.split().str.len()
    test_en_len = test_df['eng_clean'].str.split().str.len()
    
    axes[0, 0].hist(train_en_len, bins=30, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 0].hist(dev_en_len, bins=30, alpha=0.7, label='Dev', color='green', edgecolor='black')
    axes[0, 0].hist(test_en_len, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Words')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('English Sentence Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # English statistics
    axes[0, 1].axis('off')
    stats_text = f"English Length Statistics:\n\n"
    stats_text += f"Train: Mean={train_en_len.mean():.1f}, Std={train_en_len.std():.1f}, Max={train_en_len.max()}\n"
    stats_text += f"Dev:   Mean={dev_en_len.mean():.1f}, Std={dev_en_len.std():.1f}, Max={dev_en_len.max()}\n"
    stats_text += f"Test:  Mean={test_en_len.mean():.1f}, Std={test_en_len.std():.1f}, Max={test_en_len.max()}"
    axes[0, 1].text(0.1, 0.5, stats_text, transform=axes[0, 1].transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Bari lengths
    train_ba_len = train_df['bari_clean'].str.split().str.len()
    dev_ba_len = dev_df['bari_clean'].str.split().str.len()
    test_ba_len = test_df['bari_clean'].str.split().str.len()
    
    axes[1, 0].hist(train_ba_len, bins=30, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[1, 0].hist(dev_ba_len, bins=30, alpha=0.7, label='Dev', color='green', edgecolor='black')
    axes[1, 0].hist(test_ba_len, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Words')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Bari Sentence Length Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bari statistics
    axes[1, 1].axis('off')
    stats_text = f"Bari Length Statistics:\n\n"
    stats_text += f"Train: Mean={train_ba_len.mean():.1f}, Std={train_ba_len.std():.1f}, Max={train_ba_len.max()}\n"
    stats_text += f"Dev:   Mean={dev_ba_len.mean():.1f}, Std={dev_ba_len.std():.1f}, Max={dev_ba_len.max()}\n"
    stats_text += f"Test:  Mean={test_ba_len.mean():.1f}, Std={test_ba_len.std():.1f}, Max={test_ba_len.max()}"
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved length distribution to {save_path}")


def plot_training_progress(train_losses: List[float], val_losses: List[float],
                           learning_rates: Optional[List[float]] = None,
                           save_path: str = 'figures/training_progress.png'):
    """
    Plot comprehensive training progress with multiple metrics
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        learning_rates: Optional list of learning rates per epoch
        save_path: Path to save the figure
    """
    set_plot_style()
    
    if learning_rates is None:
        learning_rates = [0.001] * len(train_losses)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0, 0].plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learning rate schedule
    axes[0, 1].plot(epochs, learning_rates, 'g-d', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Loss improvement per epoch
    train_improvement = [0] + [train_losses[i-1] - train_losses[i] for i in range(1, len(train_losses))]
    val_improvement = [0] + [val_losses[i-1] - val_losses[i] for i in range(1, len(val_losses))]
    
    axes[1, 0].bar(epochs, train_improvement, alpha=0.7, label='Train Improvement', color='blue', edgecolor='black')
    axes[1, 0].bar(epochs, val_improvement, alpha=0.7, label='Val Improvement', color='red', edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Improvement')
    axes[1, 0].set_title('Loss Improvement per Epoch (Positive = Better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Best epoch indicator
    best_epoch = val_losses.index(min(val_losses)) + 1
    axes[1, 1].axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    {'='*30}
    
    Total Epochs: {len(train_losses)}
    Best Epoch: {best_epoch}
    
    Initial Train Loss: {train_losses[0]:.4f}
    Final Train Loss: {train_losses[-1]:.4f}
    Improvement: {train_losses[0] - train_losses[-1]:.4f}
    
    Initial Val Loss: {val_losses[0]:.4f}
    Best Val Loss: {min(val_losses):.4f}
    Final Val Loss: {val_losses[-1]:.4f}
    
    {'⚠️ Overfitting detected' if val_losses[-1] > min(val_losses) else '✓ No overfitting detected'}
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training progress to {save_path}")


def plot_vocabulary_comparison(whitespace_vocab: int, bpe_4k_vocab: int, bpe_8k_vocab: int,
                                oov_rates: Dict[str, float],
                                save_path: str = 'figures/vocabulary_comparison.png'):
    """
    Plot vocabulary size and OOV rate comparison
    
    Args:
        whitespace_vocab: Whitespace vocabulary size
        bpe_4k_vocab: BPE 4k vocabulary size
        bpe_8k_vocab: BPE 8k vocabulary size
        oov_rates: Dictionary with OOV rates for each method
        save_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['Whitespace', 'BPE (4k)', 'BPE (8k)']
    vocab_sizes = [whitespace_vocab, bpe_4k_vocab, bpe_8k_vocab]
    oov_values = [oov_rates.get('whitespace', 0), oov_rates.get('bpe_4k', 0), oov_rates.get('bpe_8k', 0)]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # Plot 1: Vocabulary size (log scale)
    bars1 = axes[0].bar(methods, vocab_sizes, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Vocabulary Size')
    axes[0].set_title('Vocabulary Size Comparison')
    axes[0].set_yscale('log')
    
    for bar, size in zip(bars1, vocab_sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{size:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: OOV rate
    bars2 = axes[1].bar(methods, oov_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('OOV Rate (%)')
    axes[1].set_title('Out-of-Vocabulary Rate Comparison')
    axes[1].set_ylim(0, max(oov_values) * 1.2)
    
    for bar, rate in zip(bars2, oov_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{rate:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved vocabulary comparison to {save_path}")


def create_evaluation_summary(results: Dict, output_path: str = 'figures/evaluation_summary.png'):
    """
    Create a comprehensive evaluation summary figure
    
    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the figure
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    strategies = list(results.keys())
    bleu_scores = [results[s]['bleu'] for s in strategies]
    chrf_scores = [results[s]['chrf'] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    # Grouped bar chart
    bars1 = axes[0].bar(x - width/2, bleu_scores, width, label='BLEU', color='#2E86AB', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, chrf_scores, width, label='chrF', color='#F18F01', edgecolor='black')
    
    axes[0].set_xlabel('Decoding Strategy')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Evaluation Scores by Decoding Strategy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    
    # Create summary table
    axes[1].axis('tight')
    axes[1].axis('off')
    
    table_data = []
    table_data.append(['Strategy', 'BLEU', 'chrF', 'Improvement'])
    for s in strategies:
        if s == 'greedy':
            table_data.append([s.upper(), f"{results[s]['bleu']:.2f}", f"{results[s]['chrf']:.2f}", '-'])
        else:
            improvement = results[s]['bleu'] - results['greedy']['bleu']
            table_data.append([s.upper(), f"{results[s]['bleu']:.2f}", f"{results[s]['chrf']:.2f}", f"+{improvement:.2f}"])
    
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Performance Summary Table', fontsize=12, pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved evaluation summary to {output_path}")