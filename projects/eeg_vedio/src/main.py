# This file serves as the entry point for the application. It initializes the training process and sets up the necessary configurations.

import argparse
from training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Contrastive EEG Video Fine-Tuning")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the preprocessed data')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model checkpoints and logs')

    args = parser.parse_args()

    train_model(args)

if __name__ == "__main__":
    main()