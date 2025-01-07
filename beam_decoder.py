import torch
import torch.nn.functional as F
from typing import List, Dict
import numpy as np

def load_vocab(vocab_path: str) -> Dict[int, str]:
    """Load vocabulary from file and create token-to-word mapping.
    
    Args:
        vocab_path: Path to vocabulary file
        
    Returns:
        Dictionary mapping token IDs to words
    """
    vocab = {}
    with open(vocab_path, 'r') as f:
        for idx, line in enumerate(f):
            word = line.strip()
            vocab[idx] = word
    return vocab

def tokens_to_words(tokens: List[int], vocab: Dict[int, str]) -> List[str]:
    """Convert token IDs to words using the vocabulary.
    
    Args:
        tokens: List of token IDs
        vocab: Dictionary mapping token IDs to words
        
    Returns:
        List of words
    """
    words = []
    current_word = []
    
    for token in tokens:
        if token in vocab:
            word = vocab[token]
            if word == '<space>' or word == ' ':
                if current_word:
                    words.append(''.join(current_word))
                    current_word = []
            else:
                current_word.append(word)
    
    if current_word:
        words.append(''.join(current_word))
    
    return words

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
        """
        Perform beam search decoding
        
        Args:
            audio_features: Tensor of shape (1, sequence_length, feature_dim)
            
        Returns:
            List of decoded transcriptions
        """
        batch_size = audio_features.size(0)
        
        # Get initial logits from acoustic model
        with torch.no_grad():
            logits = self.acoustic_model(audio_features)
            log_probs = F.log_softmax(logits, dim=-1)
        
        # Initialize beam
        beam_scores = torch.zeros((batch_size, self.beam_size), device=self.device)
        beam_sequences = torch.full(
            (batch_size, self.beam_size, 1),
            fill_value=0,  # Start token
            dtype=torch.long,
            device=self.device
        )
        
        # Expand first step
        vocab_size = log_probs.size(-1)
        scores, indices = log_probs[:, 0].topk(self.beam_size, dim=-1)
        beam_sequences = torch.cat([beam_sequences, indices.unsqueeze(-1)], dim=-1)
        beam_scores = scores
        
        # Iterate until max length
        for step in range(2, self.max_len):
            # Get acoustic model scores
            current_log_probs = log_probs[:, step-1] if step-1 < log_probs.size(1) else log_probs[:, -1]
            
            # Get language model scores
            lm_scores = self.language_model.score_sequence(beam_sequences)[:, -1]
            
            # Combine scores
            combined_scores = beam_scores.unsqueeze(-1) + \
                            current_log_probs + \
                            self.lm_weight * lm_scores
            
            # Add word score at word boundaries
            word_boundary_mask = (beam_sequences[:, :, -1] == self.vocab['<space>'])
            combined_scores = combined_scores + (word_boundary_mask.unsqueeze(-1) * self.word_score)
            
            # Select top-k
            flat_scores = combined_scores.view(batch_size, -1)
            scores, indices = flat_scores.topk(self.beam_size, dim=-1)
            beam_indices = indices // vocab_size
            token_indices = indices % vocab_size
            
            # Update sequences
            new_sequences = torch.cat([
                beam_sequences.gather(1, beam_indices.unsqueeze(-1).expand(-1, -1, step)),
                token_indices.unsqueeze(-1)
            ], dim=-1)
            
            # Update beam
            beam_sequences = new_sequences
            beam_scores = scores
            
            # Early stopping if all sequences end with space
            if (beam_sequences[:, 0, -1] == self.vocab['<space>']).all():
                break
        
        # Convert best sequences to words
        transcriptions = []
        for batch_idx in range(batch_size):
            best_sequence = beam_sequences[batch_idx, 0].tolist()  # Take best beam
            words = tokens_to_words(best_sequence, self.vocab)
            transcriptions.append(' '.join(words))
        
        return transcriptions 