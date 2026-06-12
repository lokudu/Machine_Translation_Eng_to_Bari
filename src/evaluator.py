# src/evaluator.py
import torch
import sacrebleu
from .beam_search import beam_search_decode
from .helpers import tokenize_english

class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def compute_bleu(self, predictions, references):
        return sacrebleu.corpus_bleu(predictions, [references]).score
    
    def compute_chrf(self, predictions, references):
        return sacrebleu.corpus_chrf(predictions, [references]).score
    
    def evaluate(self, dataset, beam_width=5, max_samples=500):
        predictions = []
        references = []
        num_samples = min(max_samples, len(dataset))
        for i in range(num_samples):
            src_text = dataset.df.iloc[i]['eng_clean']
            ref_text = dataset.df.iloc[i]['bari_clean']
            eng_ids = tokenize_english(src_text)
            src = torch.tensor([eng_ids], device=self.device)
            src_lens = torch.tensor([len(eng_ids)])
            pred_ids = beam_search_decode(self.model, src, src_lens, beam_width, device=self.device)
            pred_text = self.tokenizer.decode(pred_ids)
            predictions.append(pred_text)
            references.append(ref_text)
            if (i+1) % 100 == 0:
                print(f"  Evaluated {i+1}/{num_samples}")
        bleu = self.compute_bleu(predictions, references)
        chrf = self.compute_chrf(predictions, references)
        return bleu, chrf, predictions, references