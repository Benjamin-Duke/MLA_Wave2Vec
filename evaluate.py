import torch
import argparse
from datasets import load_dataset
import editdistance
from tqdm import tqdm
from Modules.wav2vec2_model import Wav2Vec2ForCTC
from transformer_lm import TransformerLM
from beam_decoder import BeamSearchDecoder, load_vocab

def calculate_wer(predictions: list, references: list) -> float:
    """Calculate Word Error Rate between predicted and reference texts."""
    total_wer = 0
    total_words = 0
    
    for pred, ref in zip(predictions, references):
        # Split into words
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Calculate edit distance
        distance = editdistance.eval(pred_words, ref_words)
        
        # Update totals
        total_wer += distance
        total_words += len(ref_words)
    
    return (total_wer / total_words) * 100 if total_words > 0 else 0

def evaluate_on_dataset(dataset, model, decoder, device):
    predictions = []
    references = []
    
    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataset):
            # Process audio to features
            audio = torch.tensor(batch["audio"]["array"]).unsqueeze(0).to(device)
            
            # Get transcription through the whole pipeline
            transcription = decoder.decode(audio)
            
            # Store prediction and reference
            predictions.append(transcription)
            references.append(batch["text"])
            
            # Calculate WER for current batch
            if len(predictions) % 100 == 0:
                current_wer = calculate_wer(predictions, references)
                print(f"Current WER after {len(predictions)} samples: {current_wer:.2f}%")
    
    return calculate_wer(predictions, references), predictions, references

def evaluate_model(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load fine-tuned model
    model = Wav2Vec2ForCTC.from_pretrained(args.model_path).to(device)
    model.eval()
    
    # Load language model
    print("Loading language model...")
    checkpoint = torch.load(args.lm_path, map_location=device)
    language_model = TransformerLM(checkpoint['config']).to(device)
    language_model.load_state_dict(checkpoint['model_state_dict'])
    language_model.eval()
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab(args.vocab_path)
    
    # Create beam search decoder
    decoder = BeamSearchDecoder(
        acoustic_model=model,
        language_model=language_model,
        vocab=vocab,
        beam_size=args.beam_size,
        lm_weight=args.lm_weight,
        word_score=args.word_score
    )
    
    # Load test datasets
    test_clean = load_dataset("librispeech_asr", "clean", split="test")
    test_other = load_dataset("librispeech_asr", "other", split="test")
    
    # Evaluate on test-clean
    print("\nEvaluating on test-clean...")
    wer_clean, pred_clean, ref_clean = evaluate_on_dataset(test_clean, model, decoder, device)
    print(f"Test-Clean WER: {wer_clean:.2f}%")
    
    # Evaluate on test-other
    print("\nEvaluating on test-other...")
    wer_other, pred_other, ref_other = evaluate_on_dataset(test_other, model, decoder, device)
    print(f"Test-Other WER: {wer_other:.2f}%")
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write(f"Test-Clean WER: {wer_clean:.2f}%\n")
        f.write(f"Test-Other WER: {wer_other:.2f}%\n\n")
        
        f.write("=== Sample Predictions (test-clean) ===\n")
        for pred, ref in zip(pred_clean[:5], ref_clean[:5]):
            f.write(f"Predicted: {pred}\n")
            f.write(f"Reference: {ref}\n")
            f.write("-" * 50 + "\n")
        
        f.write("\n=== Sample Predictions (test-other) ===\n")
        for pred, ref in zip(pred_other[:5], ref_other[:5]):
            f.write(f"Predicted: {pred}\n")
            f.write(f"Reference: {ref}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ASR model using WER')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the fine-tuned model')
    parser.add_argument('--lm_path', type=str, required=True,
                        help='Path to the trained language model')
    parser.add_argument('--vocab_path', type=str, default='LM_data/librispeech-vocab.txt',
                        help='Path to vocabulary file')
    parser.add_argument('--beam_size', type=int, default=100,
                        help='Beam size for decoding')
    parser.add_argument('--lm_weight', type=float, default=0.3,
                        help='Weight for language model scores')
    parser.add_argument('--word_score', type=float, default=-1.0,
                        help='Score added at word boundaries')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt',
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    evaluate_model(args) 