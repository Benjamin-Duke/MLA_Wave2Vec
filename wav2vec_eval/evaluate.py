import torch
import argparse
import editdistance
from tqdm import tqdm
import soundfile as sf
import os
import numpy as np
from src.models.wav2vec_finetuning import Wav2Vec2ForCTC
from transformer_lm import TransformerLM, TransformerLMConfig, LambdaLM
from beam_decoder import BeamSearchDecoder, load_vocab
from src.config.model_config import Wav2Vec2Config

class LocalLibriSpeechDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        for speaker_id in os.listdir(root_dir):
            speaker_dir = os.path.join(root_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                if not os.path.isdir(chapter_dir):
                    continue
                trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
                if os.path.exists(trans_file):
                    with open(trans_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            file_id = parts[0]
                            text = ' '.join(parts[1:])
                            audio_path = os.path.join(chapter_dir, f"{file_id}.flac")
                            if os.path.exists(audio_path):
                                self.samples.append({'audio_path': audio_path, 'text': text})
        print(f"Found {len(self.samples)} samples in {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample['audio_path'])
        assert sr == 16000
        return {'audio': {'array': audio.astype(np.float32), 'sampling_rate': sr}, 'text': sample['text']}

def calculate_wer(predictions: list, references: list) -> float:
    total_wer = 0
    total_words = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split() if isinstance(pred, str) else pred
        ref_words = ref.split()
        distance = editdistance.eval(pred_words, ref_words)
        total_wer += distance
        total_words += len(ref_words)
    return (total_wer / total_words) * 100 if total_words > 0 else 0

def evaluate_on_dataset(dataset, model, decoder, device):
    predictions = []
    references = []
    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataset):
            audio = torch.tensor(batch["audio"]["array"]).unsqueeze(0).to(device)
            transcription = decoder.decode(audio)
            if isinstance(transcription, list):
                predictions.append(transcription[0]) if len(transcription) == 1 else predictions.extend(transcription)
            else:
                predictions.append(transcription)
            references.append(batch["text"])
            if len(predictions) % 100 == 0:
                print(f"Current WER after {len(predictions)} samples: {calculate_wer(predictions, references):.2f}%")
    return calculate_wer(predictions, references), predictions, references

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    config = Wav2Vec2Config()
    model = Wav2Vec2ForCTC(config).to(device)
    c = torch.load(args.model_path, map_location=device)
    model.load_state_dict(c['model_state_dict'])
    model.eval()
    acoustic_vocab_size = 27
    print(f"Acoustic model vocabulary size: {acoustic_vocab_size}")
    print("Loading vocabulary...")
    vocab = load_vocab(args.vocab_path)
    actual_beam_size = min(args.beam_size, acoustic_vocab_size)
    if actual_beam_size < args.beam_size:
        print(f"Warning: Reducing beam size from {args.beam_size} to {actual_beam_size}")
    print("Creating lambda language model...")
    language_model = LambdaLM(acoustic_vocab_size).to(device)
    decoder = BeamSearchDecoder(
        acoustic_model=model,
        language_model=language_model,
        vocab=vocab,
        beam_size=actual_beam_size,
        lm_weight=args.lm_weight,
        word_score=args.word_score
    )
    test_clean = LocalLibriSpeechDataset(os.path.join(args.data_dir, 'test-clean'))
    print("\nEvaluating on test-clean...")
    wer_clean, pred_clean, ref_clean = evaluate_on_dataset(test_clean, model, decoder, device)
    print(f"Test-Clean WER: {wer_clean:.2f}%")
    with open(args.output_file, 'w') as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write(f"Test-Clean WER: {wer_clean:.2f}%\n")
        f.write("=== Sample Predictions (test-clean) ===\n")
        for pred, ref in zip(pred_clean[:5], ref_clean[:5]):
            f.write(f"Predicted: {pred}\n")
            f.write(f"Reference: {ref}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ASR model using WER')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model')
    parser.add_argument('--lm_path', type=str, required=True, help='Path to the trained language model')
    parser.add_argument('--vocab_path', type=str, default='LM_data/librispeech-vocab.txt', help='Path to vocabulary file')
    parser.add_argument('--beam_size', type=int, default=100, help='Beam size for decoding')
    parser.add_argument('--lm_weight', type=float, default=0.3, help='Weight for language model scores')
    parser.add_argument('--word_score', type=float, default=-1.0, help='Score added at word boundaries')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='Path to save evaluation results')
    parser.add_argument('--data_dir', type=str, default='', help='Path to the LibriSpeech data directory')
    parser.add_argument('--cache_dir', type=str, default='', help='Path to the cache directory')
    args = parser.parse_args()
    evaluate_model(args)
