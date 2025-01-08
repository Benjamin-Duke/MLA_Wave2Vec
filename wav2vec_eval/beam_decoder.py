import torch
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

def load_vocab(vocab_path: str) -> Dict[int, str]:
    vocab = {}
    vocab[0] = '<blank>'
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz', start=1):
        vocab[i] = c
    assert len(vocab) == 27
    return vocab

def tokens_to_words(tokens: List[int], vocab: Dict[int, str]) -> List[str]:
    chars = []
    for t in tokens:
        if t == 0:
            if chars and chars[-1] != ' ':
                chars.append(' ')
        elif t in vocab and vocab[t] != '<blank>':
            chars.append(vocab[t])
    text = ''.join(chars).strip()
    words = text.split()
    return words if words else ['']

class BeamSearchDecoder:
    def __init__(
        self,
        acoustic_model,
        language_model,
        vocab: Dict[int, str],
        beam_size: int = 100,
        lm_weight: float = 0.3,
        word_score: float = -1.0,
        max_len: int = 200
    ):
        self.acoustic_model = acoustic_model
        self.language_model = language_model
        self.vocab = vocab
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.max_len = max_len
        self.device = next(acoustic_model.parameters()).device
    
    def decode(self, audio_features: torch.Tensor) -> List[str]:
        batch_size = audio_features.size(0)
        with torch.no_grad():
            logits = self.acoustic_model(audio_features)
            log_probs = F.log_softmax(logits, dim=-1)
        vocab_size = log_probs.size(-1)
        actual_beam_size = min(self.beam_size, vocab_size)
        beam_scores = torch.zeros((batch_size, actual_beam_size), device=self.device)
        beam_sequences = torch.full(
            (batch_size, actual_beam_size, 1),
            fill_value=0,
            dtype=torch.long,
            device=self.device
        )
        scores, indices = log_probs[:, 0].topk(actual_beam_size, dim=-1)
        beam_sequences = torch.cat([beam_sequences, indices.unsqueeze(-1)], dim=-1)
        beam_scores = scores
        for step in range(2, self.max_len):
            current_log_probs = log_probs[:, step-1] if step-1 < log_probs.size(1) else log_probs[:, -1]
            with torch.no_grad():
                try:
                    lm_scores = self.language_model.score_sequence(beam_sequences)
                    lm_scores = lm_scores[:, -1]
                except RuntimeError as e:
                    print(f"Debug - beam_sequences shape: {beam_sequences.shape}")
                    print(f"Debug - d_model: {self.language_model.config.d_model}")
                    raise e
            combined_scores = beam_scores.unsqueeze(-1) + \
                            current_log_probs + \
                            self.lm_weight * lm_scores
            word_boundary_mask = (beam_sequences[:, :, -1] == 0)
            combined_scores = combined_scores + (word_boundary_mask.unsqueeze(-1) * self.word_score)
            flat_scores = combined_scores.view(batch_size, -1)
            scores, indices = flat_scores.topk(actual_beam_size, dim=-1)
            beam_indices = indices // vocab_size
            token_indices = indices % vocab_size
            new_sequences = torch.cat([
                beam_sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, step)),
                token_indices.unsqueeze(-1)
            ], dim=-1)
            beam_sequences = new_sequences
            beam_scores = scores
            if (beam_sequences[:, 0, -1] == 0).all():
                break
        transcriptions = []
        for batch_idx in range(batch_size):
            best_sequence = beam_sequences[batch_idx, 0].tolist()
            words = tokens_to_words(best_sequence, self.vocab)
            transcriptions.append(' '.join(words))
        return transcriptions
