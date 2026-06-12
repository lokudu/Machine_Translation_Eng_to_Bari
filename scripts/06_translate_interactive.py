#!/usr/bin/env python
# scripts/06_translate_interactive.py
import sys
import torch
sys.path.append('..')
from src.config import PATH_CONFIG
from src.seq2seq import Seq2Seq
from src.bpe_tokenizer import BPETokenizer
from src.beam_search import beam_search_decode
from src.helpers import tokenize_english, get_device

def translate_sentence(model, tokenizer, sentence, beam_width=5, device='cuda'):
    eng_ids = tokenize_english(sentence)
    if not eng_ids:
        return ""
    src = torch.tensor([eng_ids], device=device)
    src_lens = torch.tensor([len(eng_ids)])
    pred_ids = beam_search_decode(model, src, src_lens, beam_width, device=device)
    return tokenizer.decode(pred_ids)

def main():
    print("="*60)
    print("INTERACTIVE TRANSLATION DEMO")
    print("English to Bari")
    print("="*60)
    device = get_device()
    print("\nLoading model...")
    tokenizer = BPETokenizer()
    tokenizer.load(f"{PATH_CONFIG.models_dir}bpe_8k.model")
    model = Seq2Seq(
        src_vocab=30000,
        tgt_vocab=tokenizer.get_vocab_size(),
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    )
    model.load_state_dict(torch.load(f"{PATH_CONFIG.models_dir}best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    print("\nEnter English sentences (type 'quit' to exit):")
    print("-"*40)
    while True:
        sentence = input("\nEnglish: ").strip()
        if sentence.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        if not sentence:
            continue
        translation = translate_sentence(model, tokenizer, sentence, beam_width=5, device=device)
        print(f"Bari: {translation if translation else '[No translation]'}")

if __name__ == "__main__":
    main()