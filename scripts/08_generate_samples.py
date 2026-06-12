#!/usr/bin/env python
# scripts/08_generate_samples.py
"""
Generate 20 sample translations and save to results folder
Includes source text, reference translation, model outputs, and evaluation
"""

import sys
import os
import pandas as pd
import random
from datetime import datetime

# Add project root to path
project_root = r'C:\Users\Lokudu James\english-bari-mt'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.config import PATH_CONFIG
from src.dataset import BariTranslationDataset
from src.seq2seq import Seq2Seq
from src.bpe_tokenizer import BPETokenizer
from src.beam_search import beam_search_decode
from src.helpers import tokenize_english, get_device
from src.greedy import greedy_decode


def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    device = get_device()
    
    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(PATH_CONFIG.models_dir, "bpe_8k.model"))
    
    model = Seq2Seq(
        src_vocab=30000,
        tgt_vocab=tokenizer.get_vocab_size(),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    )
    model_path = os.path.join(PATH_CONFIG.models_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"✓ Loaded tokenizer from {PATH_CONFIG.models_dir}")
    print(f"  Device: {device}")
    
    return model, tokenizer, device


def translate_sentence(model, tokenizer, sentence, beam_width=5, device='cuda'):
    """Translate a single sentence"""
    eng_ids = tokenize_english(sentence)
    if not eng_ids:
        return ""
    
    src = torch.tensor([eng_ids], device=device)
    src_lens = torch.tensor([len(eng_ids)])
    
    if beam_width == 1:
        pred_ids = greedy_decode(model, src, src_lens, device=device)
    else:
        pred_ids = beam_search_decode(model, src, src_lens, beam_width, device=device)
    
    return tokenizer.decode(pred_ids)


def generate_samples(test_df, model, tokenizer, device, num_samples=20):
    """Generate sample translations from test set"""
    
    # Randomly sample sentences from test set
    if len(test_df) > num_samples:
        sample_df = test_df.sample(n=num_samples, random_state=42)
    else:
        sample_df = test_df
    
    samples = []
    
    print(f"\nGenerating {len(sample_df)} sample translations...")
    
    for idx, row in sample_df.iterrows():
        source = row['eng_clean']
        reference = row['bari_clean']
        
        # Generate with different decoding strategies
        greedy_trans = translate_sentence(model, tokenizer, source, beam_width=1, device=device)
        beam3_trans = translate_sentence(model, tokenizer, source, beam_width=3, device=device)
        beam5_trans = translate_sentence(model, tokenizer, source, beam_width=5, device=device)
        
        # Calculate lengths
        src_len = len(source.split())
        ref_len = len(reference.split())
        greedy_len = len(greedy_trans.split()) if greedy_trans else 0
        beam3_len = len(beam3_trans.split()) if beam3_trans else 0
        beam5_len = len(beam5_trans.split()) if beam5_trans else 0
        
        samples.append({
            'id': idx + 1,
            'source': source,
            'reference': reference,
            'greedy': greedy_trans if greedy_trans else '[EMPTY]',
            'beam3': beam3_trans if beam3_trans else '[EMPTY]',
            'beam5': beam5_trans if beam5_trans else '[EMPTY]',
            'src_len': src_len,
            'ref_len': ref_len,
            'greedy_len': greedy_len,
            'beam3_len': beam3_len,
            'beam5_len': beam5_len
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(sample_df)} samples...")
    
    return samples


def save_samples_to_file(samples, output_path):
    """Save samples to a formatted text file"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("ENGLISH TO BARI TRANSLATION - SAMPLE OUTPUTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {len(samples)}\n")
        f.write("=" * 100 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        
        avg_src_len = sum(s['src_len'] for s in samples) / len(samples)
        avg_ref_len = sum(s['ref_len'] for s in samples) / len(samples)
        avg_greedy_len = sum(s['greedy_len'] for s in samples) / len(samples)
        avg_beam3_len = sum(s['beam3_len'] for s in samples) / len(samples)
        avg_beam5_len = sum(s['beam5_len'] for s in samples) / len(samples)
        
        f.write(f"Average Source Length:      {avg_src_len:.1f} words\n")
        f.write(f"Average Reference Length:   {avg_ref_len:.1f} words\n")
        f.write(f"Average Greedy Length:      {avg_greedy_len:.1f} words\n")
        f.write(f"Average Beam-3 Length:      {avg_beam3_len:.1f} words\n")
        f.write(f"Average Beam-5 Length:      {avg_beam5_len:.1f} words\n\n")
        
        # Table format
        f.write("=" * 120 + "\n")
        f.write("SAMPLE TRANSLATIONS TABLE\n")
        f.write("=" * 120 + "\n\n")
        
        # Header row
        f.write(f"{'ID':<4} | {'Source (EN)':<40} | {'Reference (BA)':<35} | {'Greedy':<20} | {'Beam-5':<20}\n")
        f.write("-" * 120 + "\n")
        
        for sample in samples:
            # Truncate long texts
            src = sample['source'][:37] + "..." if len(sample['source']) > 40 else sample['source']
            ref = sample['reference'][:32] + "..." if len(sample['reference']) > 35 else sample['reference']
            greedy = sample['greedy'][:17] + "..." if len(sample['greedy']) > 20 else sample['greedy']
            beam5 = sample['beam5'][:17] + "..." if len(sample['beam5']) > 20 else sample['beam5']
            
            f.write(f"{sample['id']:<4} | {src:<40} | {ref:<35} | {greedy:<20} | {beam5:<20}\n")
        
        f.write("\n\n")
        
        # Detailed samples
        f.write("=" * 100 + "\n")
        f.write("DETAILED SAMPLES (FULL TEXT)\n")
        f.write("=" * 100 + "\n")
        
        for sample in samples:
            f.write(f"\n{'='*80}\n")
            f.write(f"SAMPLE {sample['id']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Source (EN):      {sample['source']}\n")
            f.write(f"Reference (BA):   {sample['reference']}\n")
            f.write(f"\nTranslations:\n")
            f.write(f"  Greedy:   {sample['greedy']}\n")
            f.write(f"  Beam-3:   {sample['beam3']}\n")
            f.write(f"  Beam-5:   {sample['beam5']}\n")
            f.write(f"\nLength (words):\n")
            f.write(f"  Source: {sample['src_len']} | Reference: {sample['ref_len']}\n")
            f.write(f"  Greedy: {sample['greedy_len']} | Beam-3: {sample['beam3_len']} | Beam-5: {sample['beam5_len']}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF SAMPLE TRANSLATIONS\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")


def save_samples_to_csv(samples, output_path):
    """Save samples to CSV for Excel analysis"""
    
    import csv
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['ID', 'Source_EN', 'Reference_BA', 'Greedy_Output', 'Beam3_Output', 'Beam5_Output',
                         'Source_Len', 'Ref_Len', 'Greedy_Len', 'Beam3_Len', 'Beam5_Len'])
        
        for sample in samples:
            writer.writerow([
                sample['id'],
                sample['source'],
                sample['reference'],
                sample['greedy'],
                sample['beam3'],
                sample['beam5'],
                sample['src_len'],
                sample['ref_len'],
                sample['greedy_len'],
                sample['beam3_len'],
                sample['beam5_len']
            ])
    
    print(f"✓ Saved CSV to {output_path}")


def save_samples_to_html(samples, output_path):
    """Save samples to HTML for better viewing"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>English to Bari Translation Samples</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .sample {{ margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; }}
            .sample-title {{ background-color: #3498db; color: white; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 5px 5px 0 0; }}
            .english {{ color: #2980b9; font-weight: bold; }}
            .bari {{ color: #27ae60; font-weight: bold; }}
            .translation {{ margin-left: 20px; }}
            .footer {{ margin-top: 30px; text-align: center; font-size: 12px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <h1>English to Bari Translation - Sample Outputs</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Samples: {len(samples)}</p>
        
        <div class="summary">
            <h3>Summary Statistics</h3>
            <p>Average Source Length: {sum(s['src_len'] for s in samples)/len(samples):.1f} words</p>
            <p>Average Reference Length: {sum(s['ref_len'] for s in samples)/len(samples):.1f} words</p>
            <p>Average Greedy Length: {sum(s['greedy_len'] for s in samples)/len(samples):.1f} words</p>
            <p>Average Beam-5 Length: {sum(s['beam5_len'] for s in samples)/len(samples):.1f} words</p>
        </div>
        
        <h2>Translation Table</h2>
        <table>
            <tr>
                <th>ID</th>
                <th>Source (English)</th>
                <th>Reference (Bari)</th>
                <th>Greedy Output</th>
                <th>Beam-5 Output</th>
            </tr>
    """
    
    for sample in samples:
        html_content += f"""
            <tr>
                <td>{sample['id']}</td>
                <td>{sample['source'][:100]}{'...' if len(sample['source']) > 100 else ''}</td>
                <td>{sample['reference'][:100]}{'...' if len(sample['reference']) > 100 else ''}</td>
                <td>{sample['greedy'][:80]}{'...' if len(sample['greedy']) > 80 else ''}</td>
                <td>{sample['beam5'][:80]}{'...' if len(sample['beam5']) > 80 else ''}</td>
            </tr>
        """
    
    html_content += """
        <table>
        
        <h2>Detailed Samples</h2>
    """
    
    for sample in samples:
        html_content += f"""
        <div class="sample">
            <div class="sample-title">Sample {sample['id']}</div>
            <div class="english">📖 English: {sample['source']}</div>
            <div class="bari">📖 Reference Bari: {sample['reference']}</div>
            <div class="translation">
                <p><strong>Greedy Translation:</strong> {sample['greedy']}</p>
                <p><strong>Beam-3 Translation:</strong> {sample['beam3']}</p>
                <p><strong>Beam-5 Translation:</strong> {sample['beam5']}</p>
                <p><strong>Lengths:</strong> Source={sample['src_len']}, Reference={sample['ref_len']}, Greedy={sample['greedy_len']}, Beam-5={sample['beam5_len']}</p>
            </div>
        </div>
        """
    
    html_content += """
        <div class="footer">
            <p>Generated by English-Bari Neural Machine Translation System</p>
            <p>MSA 7202: Natural Language Processing Project</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Saved HTML to {output_path}")


def main():
    print("="*60)
    print("GENERATING SAMPLE TRANSLATIONS (20 SAMPLES)")
    print("="*60)
    
    # Load test data
    test_path = os.path.join(PATH_CONFIG.data_dir, "processed", "test.csv")
    
    if not os.path.exists(test_path):
        print(f"\n✗ Test file not found at {test_path}")
        print("Please run scripts/02_preprocess_data.py first.")
        return
    
    test_df = pd.read_csv(test_path)
    print(f"✓ Loaded test data: {len(test_df)} pairs")
    
    # Load model and tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer()
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        return
    
    # Generate samples
    samples = generate_samples(test_df, model, tokenizer, device, num_samples=20)
    
    # Save to various formats
    os.makedirs(PATH_CONFIG.results_dir, exist_ok=True)
    
    # Text file
    txt_path = os.path.join(PATH_CONFIG.results_dir, "sample_translations.txt")
    save_samples_to_file(samples, txt_path)
    
    # CSV file
    csv_path = os.path.join(PATH_CONFIG.results_dir, "sample_translations.csv")
    save_samples_to_csv(samples, csv_path)
    
    # HTML file
    html_path = os.path.join(PATH_CONFIG.results_dir, "sample_translations.html")
    save_samples_to_html(samples, html_path)
    
    # Print preview
    print("\n" + "="*60)
    print("PREVIEW OF FIRST 3 SAMPLES")
    print("="*60)
    
    for sample in samples[:3]:
        print(f"\nSample {sample['id']}:")
        print(f"  Source:   {sample['source'][:80]}...")
        print(f"  Reference: {sample['reference'][:80]}...")
        print(f"  Beam-5:    {sample['beam5'][:80]}...")
        print(f"  Lengths:   Ref={sample['ref_len']}, Beam-5={sample['beam5_len']}")
    
    print("\n" + "="*60)
    print("✓ Sample generation complete!")
    print(f"  Files saved to: {PATH_CONFIG.results_dir}")
    print("  - sample_translations.txt")
    print("  - sample_translations.csv")
    print("  - sample_translations.html")
    print("="*60)


if __name__ == "__main__":
    main()