import torch
import argparse
from transformer_lm import TransformerLM
from beam_decoder import BeamSearchDecoder
from wav2vec_finetuning import Wav2Vec2ForCTC
from Modules.config import Wav2Vec2Config

def load_models(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load acoustic model
    config = Wav2Vec2Config()
    acoustic_model = Wav2Vec2ForCTC(config)
    checkpoint = torch.load(args.acoustic_model_path, map_location=device)
    acoustic_model.load_state_dict(checkpoint['model_state_dict'])
    acoustic_model.to(device)
    acoustic_model.eval()
    
    # Load language model
    lm_checkpoint = torch.load(args.language_model_path, map_location=device)
    language_model = TransformerLM(lm_checkpoint['config'])
    language_model.load_state_dict(lm_checkpoint['model_state_dict'])
    language_model.to(device)
    language_model.eval()
    
    return acoustic_model, language_model

def decode_audio(args):
    # Load models
    acoustic_model, language_model = load_models(args)
    
    # Create decoder
    decoder = BeamSearchDecoder(
        model=acoustic_model,
        language_model=language_model,
        beam_size=args.beam_size,
        lm_weight=args.lm_weight,
        word_insertion_penalty=args.word_insertion_penalty
    )
    
    # If in optimization mode, run hyperparameter optimization
    if args.optimize:
        print("Optimizing hyperparameters...")
        best_params = decoder.optimize_hyperparameters(
            validation_data=val_dataset,  # You'll need to implement this
            n_trials=args.n_trials
        )
        print(f"Best parameters: {best_params}")
    
    # Decode test set
    print("Decoding test set...")
    total_wer = 0
    total_samples = 0
    
    with torch.no_grad():
        for audio, text in test_dataset:  # You'll need to implement this
            audio = audio.to(acoustic_model.device)
            predicted_sequence = decoder.decode(audio.unsqueeze(0))[0]
            
            # Calculate WER
            wer = calculate_wer(predicted_sequence, text)
            total_wer += wer
            total_samples += 1
            
            if total_samples % 100 == 0:
                print(f"Processed {total_samples} samples. Current WER: {total_wer/total_samples:.4f}")
    
    final_wer = total_wer / total_samples
    print(f"Final Word Error Rate: {final_wer:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decode audio using Wav2Vec2 and Transformer LM')
    
    parser.add_argument('--acoustic_model_path', type=str, required=True,
                        help='Path to acoustic model checkpoint')
    parser.add_argument('--language_model_path', type=str, required=True,
                        help='Path to language model checkpoint')
    parser.add_argument('--beam_size', type=int, default=500,
                        help='Beam size for decoding')
    parser.add_argument('--lm_weight', type=float, default=0.5,
                        help='Language model weight')
    parser.add_argument('--word_insertion_penalty', type=float, default=0.0,
                        help='Word insertion penalty')
    parser.add_argument('--optimize', action='store_true',
                        help='Whether to optimize hyperparameters')
    parser.add_argument('--n_trials', type=int, default=128,
                        help='Number of optimization trials')
    
    args = parser.parse_args()
    decode_audio(args) 