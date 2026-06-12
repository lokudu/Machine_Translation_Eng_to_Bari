# src/beam_search.py
import torch
from .greedy import apply_repetition_penalty

def apply_length_penalty(score, length, alpha=0.8):
    return score / ((length + 5) ** alpha / (6 ** alpha))

def beam_search_decode(model, src, src_lens, beam_width=5, max_len=50, repetition_penalty=1.2, length_penalty_alpha=0.8, eos_id=3, bos_id=2, device='cuda'):
    model.eval()
    with torch.no_grad():
        encoder_outputs, decoder_hidden = model.encoder(src, src_lens)
        mask = (src != 0).float()
        beams = [([bos_id], 0.0, decoder_hidden, [])]
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden, gen_ids in beams:
                if seq[-1] == eos_id:
                    all_candidates.append((seq, score, hidden, gen_ids))
                    continue
                decoder_input = torch.tensor([[seq[-1]]], device=device)
                prediction, new_hidden = model.decoder(decoder_input, hidden, encoder_outputs, mask)
                logits = prediction.squeeze()
                if repetition_penalty > 1.0 and len(gen_ids) > 0:
                    logits = apply_repetition_penalty(logits.clone(), gen_ids, repetition_penalty)
                log_probs = torch.log_softmax(logits, dim=-1)
                top_k_log_probs, top_k_indices = torch.topk(log_probs, min(beam_width, log_probs.size(-1)))
                for i in range(top_k_indices.size(0)):
                    token_id = top_k_indices[i].item()
                    token_log_prob = top_k_log_probs[i].item()
                    new_seq = seq + [token_id]
                    new_gen_ids = gen_ids + [token_id]
                    new_score = score + token_log_prob
                    penalized_score = apply_length_penalty(new_score, len(new_seq), length_penalty_alpha)
                    all_candidates.append((new_seq, penalized_score, new_hidden, new_gen_ids))
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
            if all(seq[-1] == eos_id for seq, _, _, _ in beams):
                break
        best_beam = max(beams, key=lambda x: x[1])
        return best_beam[0][1:]