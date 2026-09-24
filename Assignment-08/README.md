# Deep Learning Assignment 8: Pretrained BERT for Sentiment Analysis

## Problem Statement

Implement a pretrained BERT model for sentiment analysis using the IMDB movie review dataset. Fine-tune BERT to classify reviews as positive or negative and evaluate its performance using various classification metrics.

## Dataset Overview

- **Name:** IMDB Movie Review Dataset
- **Structure:** 50,000 labeled movie reviews, with 25,000 training and 25,000 testing samples.
- **Classes:** 2 categories (Negative, Positive).
- **Sample Used:** 2,000 training reviews and 500 testing reviews.
- **Text Data:** Movie reviews with binary sentiment labels (0 = Negative, 1 = Positive).

## Implementation Details

1. **Dataset Loading & Exploration:**
   - Loaded the IMDB dataset using Hugging Face Datasets.
   - Selected a smaller sample of training and testing reviews for efficient experimentation.
   - Displayed dataset dimensions, sample reviews, labels, and class distribution.

2. **Preprocessing & Tokenization:**
   - Loaded the pretrained `bert-base-uncased` tokenizer.
   - Converted movie reviews into BERT-compatible token IDs.
   - Applied truncation with a maximum sequence length of 256 tokens.
   - Used dynamic padding to prepare batches for model training.

3. **Pretrained BERT Model:**
   - Loaded the pretrained BERT model using Hugging Face Transformers.
   - Added a binary sequence classification head for positive and negative sentiment prediction.
   - Configured label mappings for both sentiment classes.

4. **Training & Experimentation:**
   - Fine-tuned BERT using the Hugging Face Trainer.
   - Used the AdamW optimizer with a learning rate of `2e-5`.
   - Compared model training for 1 epoch and 2 epochs.
   - Evaluated model performance using accuracy, precision, recall, and F1-score.

5. **Evaluation & Prediction:**
   - Generated a classification report on the test dataset.
   - Tested the trained model on custom movie review sentences.
   - Displayed predicted sentiment and confidence scores.
   - Saved the fine-tuned BERT model and tokenizer for future use.

## Technologies Used

- `transformers` (Pretrained BERT and Fine-Tuning)
- `datasets` (IMDB Dataset Loading)
- `torch` / `pytorch` (Deep Learning and Model Training)
- `sklearn` (Evaluation Metrics)
- `pandas` (Data Analysis and Results Handling)
- `numpy` (Numerical Processing)
