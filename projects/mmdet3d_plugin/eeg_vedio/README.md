# Contrastive EEG Video Fine-Tuning

This project implements a contrastive learning framework that aligns video data with EEG signals using a video-swin-transformer architecture. The goal is to fine-tune a video encoder to incorporate knowledge from EEG data, enabling improved performance on downstream tasks.

## Project Structure

- `src/data/preprocess.py`: Functions for preprocessing EEG and video data pairs.
- `src/models/eeg_encoder.py`: Defines the `EEGEncoder` class for encoding EEG data.
- `src/models/video_encoder.py`: Implements the `VideoEncoder` class using the video-swin-transformer model.
- `src/models/contrastive_model.py`: Combines EEG and video encoders for contrastive learning.
- `src/training/train.py`: Contains the training loop for the contrastive learning process.
- `src/utils/helpers.py`: Utility functions for data augmentation, logging, and metrics.
- `src/main.py`: Entry point for initializing the training process.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd contrastive-eeg-video-finetuning
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

To start the training process, run the following command:
```
python src/main.py
```

Make sure to adjust any configurations in `src/main.py` as needed for your specific dataset and training parameters.

## Acknowledgments

This project leverages state-of-the-art techniques in contrastive learning and deep learning for EEG and video data. Contributions and feedback are welcome!