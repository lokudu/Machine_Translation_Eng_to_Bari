# src/greedy.py
import torch

def apply_repetition_penalty(logits, generated_ids, penalty_factor=1.2):
    if not generated_ids or penalty_factor <= 1.0:
        return logits
    for token_id in set(generated_ids):
        if token_id < logits.size(0):
            if logits[token_id] > 0:
                logits[token_id] /= penalty_factor
            else:
                logits[token_id] *= penalty_factor
    return logits

def greedy_decode(model, src, src_lens, max_len=50, repetition_penalty=1.2, eos_id=3, device='cuda'):
    model.eval()
    generated_ids = []
    with torch.no_grad():
        encoder_outputs, decoder_hidden = model.encoder(src, src_lens)
        mask = (src != 0).float()
        decoder_input = torch.tensor([[2]], device=device)
        for _ in range(max_len):
            prediction, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            logits = prediction.squeeze()
            if repetition_penalty > 1.0 and len(generated_ids) > 0:
                logits = apply_repetition_penalty(logits.clone(), generated_ids, repetition_penalty)
            next_token = torch.argmax(logits).item()
            if next_token == eos_id:
                break
            generated_ids.append(next_token)
            decoder_input = torch.tensor([[next_token]], device=device)
    return generated_ids