# Speech Enhancement Project

## Overview
This project is an advanced endeavor in the field of audio processing, specifically aimed at enhancing speech quality in noisy audio recordings. Utilizing deep learning models like U-Net and convolutional autoencoders, the project addresses common challenges in audio clarity, targeting various types of environmental noise and distortions.

## Core Components

### Data Preparation and Processing
- **Scripts**:
  - `prepare_data.py`: Preprocesses audio files, including noise addition and conversion to spectrograms for deep learning compatibility.
  - `add_noise.py`: Augments audio data by blending recorded speech with various noise profiles.
- **Techniques**:
  - Spectrogram generation using `librosa`.
  - Data augmentation for robust model training.

### Model Architecture and Training
- **U-Net (`model_unet.py`, `train_model.py`)**:
  - Utilizes the U-Net architecture, renowned for its efficacy in segmentation tasks, adapted for audio denoising.
  - Training script includes model checkpoints, early stopping, and performance evaluation.
- **Convolutional Autoencoder (`autoencoder.py`, `train_model2.py`)**:
  - Employs an autoencoder architecture for feature extraction and reconstruction, aiming to isolate and remove noise components.
  - Includes custom layer configurations and training routines.

### Flask Web Application (`app.py`)
- Facilitates user interaction with the model.
- Features include file upload, audio processing, and retrieval of enhanced audio.

### Visualization and Utility Scripts
- `data_display.py`, `data_tools.py`: Provide functionalities for visualizing audio data and additional utility operations.
- `prediction_denoise.py`: Handles the prediction phase, applying trained models to denoise audio files.

## Technologies and Libraries
- **Deep Learning**: TensorFlow, Keras.
- **Audio Processing**: Librosa.
- **Data Handling and Visualization**: Numpy, Scipy, Matplotlib.
- **Web Framework**: Flask.

## Installation
1. **Environment Setup**:
   - Python 3.x and pip.
   - Install required libraries: `pip install tensorflow keras librosa flask numpy scipy matplotlib`.
2. **Repository Setup**:
   - Clone the repository: `git clone [repository-url]`.
   - Navigate to the project directory.

## Usage
1. **Data Preparation**:
   - Run `prepare_data.py` to process the raw audio files.
2. **Model Training**:
   - Execute `train_model.py` and `train_model2.py` to train the U-Net and autoencoder models, respectively.
3. **Web Application**:
   - Start the Flask app with `python app.py`.
   - Access the web interface for real-time audio processing.

