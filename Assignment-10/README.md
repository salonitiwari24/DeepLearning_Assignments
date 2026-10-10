
# Deep Learning Assignment: Speech Emotion Recognition using CNN, Whisper and gTTS

## Problem Statement

Implement a speech emotion recognition system using a Convolutional Neural Network (CNN) trained on the RAVDESS audio dataset. Extract Mel spectrogram features from speech recordings to classify eight emotions. Integrate Whisper for speech-to-text transcription and gTTS for text-to-speech generation, creating an end-to-end speech processing pipeline.

## Dataset Overview

- **Name:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Structure:** Audio recordings organized by actors, emotions, and recording conditions.
- **Classes:** 8 emotions (Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised).
- **Sample Used:** Up to 60 audio samples per emotion, with a maximum of 480 samples.
- **Audio Data:** Speech recordings in WAV format.
- **Features:** 64-bin Mel spectrograms with a fixed input shape of `(64, 128, 1)`.
- **Train-Test Split:** 80% training and 20% testing.

## Implementation Details

1. **Dataset Loading & Exploration:**
   - Uploaded and extracted the RAVDESS dataset ZIP file.
   - Collected speech audio files and extracted emotion labels from filenames.
   - Selected up to 60 samples per emotion for faster experimentation.

2. **Audio Preprocessing & Feature Extraction:**
   - Resampled audio to 22,050 Hz.
   - Standardized audio duration to 3 seconds using padding and truncation.
   - Extracted 64-bin Mel spectrograms using Librosa.
   - Converted spectrograms to decibel scale and standardized them to 128 time frames.

3. **CNN Model Development:**
   - Designed a CNN using three convolutional layers with 32, 64, and 128 filters.
   - Applied max pooling, global average pooling, and dropout layers.
   - Added dense layers with a softmax output for eight-class emotion classification.

4. **Training & Experimentation:**
   - Trained the CNN for 8 epochs using the Adam optimizer.
   - Used categorical cross-entropy as the loss function.
   - Monitored training and validation accuracy during training.

5. **Model Evaluation:**
   - Evaluated the model on the test dataset.
   - Generated a classification report containing precision, recall, and F1-score.
   - Visualized class-wise predictions using a confusion matrix.
   - Plotted training and validation accuracy to analyze model performance.

6. **Speech-to-Text using Whisper:**
   - Loaded the pretrained `openai/whisper-base` model using Hugging Face Transformers.
   - Uploaded a WAV audio file for testing.
   - Generated a text transcription of the spoken audio.

7. **Emotion Prediction & Text-to-Speech:**
   - Preprocessed the uploaded audio using the same Mel spectrogram extraction procedure.
   - Predicted the emotion and displayed the corresponding confidence score.
   - Used gTTS to convert the recognized text into speech.
   - Played the generated audio output.

8. **Model Saving:**
   - Saved the trained CNN model in Keras format.
   - Saved the emotion class labels for reuse during inference.

## Technologies Used

- `TensorFlow / Keras` (CNN Model Development and Training)
- `Librosa` (Audio Processing and Mel Spectrogram Extraction)
- `Hugging Face Transformers` (Pretrained Whisper Speech Recognition)
- `PyTorch` (Whisper Model Execution)
- `gTTS` (Text-to-Speech Generation)
- `scikit-learn` (Label Encoding and Evaluation Metrics)
- `NumPy` (Numerical Processing)
- `Matplotlib` and `Seaborn` (Performance Visualization)
- `Google Colab` (Implementation and Execution)

## Results

- **Test Accuracy:** 15.62%
- **Test Loss:** Approximately 2.076
- **Emotion Prediction:** Calm
- **Prediction Confidence:** 13.08%
- **Speech Transcription:** Successfully generated text using Whisper.
- **Text-to-Speech:** Generated and played speech audio using gTTS.

## Conclusion

The project implements an end-to-end speech processing pipeline combining CNN-based emotion classification, Whisper speech transcription, and gTTS speech generation. The initial CNN results indicate that further improvements in feature extraction, model training, and classification performance are needed for reliable emotion recognition.
