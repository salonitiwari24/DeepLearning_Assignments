# Deep Learning Assignment 5: RNN, LSTM and GRU Sequence Classification

## Problem Statement

Implement and compare **RNN, LSTM, and GRU models** for sequence classification and analyze their performance using appropriate evaluation metrics.

## Dataset Overview

- **Dataset:** IMDB Movie Reviews
- **Source:** TensorFlow/Keras
- **Samples:** 50,000 movie reviews
- **Classes:** Positive and Negative
- **Vocabulary:** 10,000 most frequent words
- **Sequence Length:** 200 tokens
- **Target:** Sentiment classification

## Implementation Details

1. **Data Loading & Preprocessing:**
   - Loaded the IMDB movie review dataset.
   - Limited the vocabulary to 10,000 words.
   - Padded all reviews to a fixed length of 200 tokens.
   - Verified the distribution of positive and negative reviews.

2. **Model Development:**
   - Developed separate **Simple RNN, LSTM, and GRU** models.
   - Used an Embedding layer with 128 dimensions.
   - Used 64 recurrent units.
   - Added a sigmoid Dense layer for binary classification.
   - Compiled the models using the Adam optimizer and Binary Crossentropy loss.

3. **Training:**
   - Trained each model for 3 epochs.
   - Used a batch size of 128.
   - Used 20% of the training data for validation.

4. **Evaluation & Comparison:**
   - Evaluated all models on the test dataset.
   - Calculated Accuracy, Precision, Recall, and F1-score.
   - Generated confusion matrices for each model.
   - Compared model performance using a bar graph.
   - Plotted validation accuracy and validation loss across epochs.

## Technologies Used

- `tensorflow` / `keras` (RNN, LSTM, GRU, Embedding, Dense)
- `numpy`
- `pandas`
- `scikit-learn` (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
- `matplotlib`
- `seaborn`
