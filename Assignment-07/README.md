# Deep Learning Assignment 7: Transfer Learning using Pre-trained Models

## Problem Statement


Implement transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification and compare their performance.

## Dataset Overview

- **Name:** CIFAR-10
- **Structure:** 60,000 color images, including 50,000 training and 10,000 testing images.
- **Classes:** 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Image Size:** 32 × 32 pixels.

## Implementation Details


1. **Dataset Loading & Exploration:**
   - Loaded the CIFAR-10 dataset using TensorFlow/Keras.
   - Displayed dataset dimensions, class names, class distribution, and sample images.

2. **Preprocessing:**
   - Resized CIFAR-10 images from `(32, 32)` to `(224, 224)` to match the input requirements of the pre-trained models.
   - Applied ImageNet-compatible normalization.
   - Created batched and prefetched TensorFlow data pipelines.

3. **Transfer Learning Models:**
   - Implemented four image classification models:
     - AlexNet
     - VGG16
     - ResNet50
     - EfficientNetB0
   - Used pre-trained ImageNet feature extractors.
   - Froze the pre-trained layers and replaced the final classification layer with a 10-class output layer.
   - Used Global Average Pooling, Dense, and Dropout layers for classification.

4. **Training & Evaluation:**
   - Trained each model for 5 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
   - Recorded test accuracy and training time for each model.
   - Compared the performance of all four models using a tabular summary and accuracy graphs.
   - Generated classification reports containing precision, recall, and F1-score.
   - Generated confusion matrices to analyze class-wise classification performance.

## Technologies Used


- `tensorflow` / `keras` (Transfer Learning and Model Training)
- `torchvision` / `pytorch` (Pre-trained model support, if applicable)
- `sklearn` (Evaluation Metrics)
- `pandas` (Results and Data Handling)
- `numpy` (Numerical Processing)
- `matplotlib` (Visualization)
- `seaborn` (Class Distribution and Confusion Matrix)
