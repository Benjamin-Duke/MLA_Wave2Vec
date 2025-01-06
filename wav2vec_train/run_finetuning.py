import torch
from torch import nn
import os
import argparse

from src.data.dataLibriSpeech import LibriSpeech
from wav2vec_finetuning import Wav2Vec2ForCTC
from src.config.model_config import Wav2Vec2Config

from src.utils.finetuning_trainer import FineTuningTrainer

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = LibriSpeech(split="dev-clean", target_length=48000)
    val_dataset = LibriSpeech(split="dev-other", target_length=48000)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load configuration
    config = Wav2Vec2Config()
    
    # Create fine-tuning model
    print("\nInitializing model...")
    model = Wav2Vec2ForCTC(config)
    
    # Load pre-trained weights
    print(f"\nLoading pre-trained weights from {args.pretrained_path}")
    checkpoint = torch.load(args.pretrained_path, map_location=device)

    model.wav2vec.load_state_dict(checkpoint['model_state_dict'])
    
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


    
    trainer = FineTuningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_steps=args.num_steps,
        classifier_only_steps=args.classifier_steps,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Load checkpoint if continuing training
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    print("\nStarting fine-tuning...")
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune Wav2Vec2 model')
    
    # Required arguments
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pre-trained model checkpoint')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='Number of training steps')
    parser.add_argument('--classifier_steps', type=int, default=10000,
                        help='Number of steps to train only the classifier')
    parser.add_argument('--log_dir', type=str, default='finetuning_runs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='finetuning_checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--resume_from', type=str,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    main(args) 








