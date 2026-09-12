# Deep Learning Assignment 1: Introduction to TensorFlow & Keras

## Problem Statement
Install and configure TensorFlow/Keras in Google Colab, and perform data preprocessing, normalization, train-test splitting, and data visualization on a sample dataset[cite: 4].

## Dataset Overview
- **Name:** Fashion MNIST[cite: 4]
- **Structure:** 60,000 training images and 10,000 testing images[cite: 4].
- **Features:** 28x28 grayscale images of clothing items[cite: 4].

## Implementation Details
1. **Data Loading & Preprocessing:** 
   - Loaded the Fashion MNIST dataset and extracted `train_images`, `train_labels`, `test_images`, and `test_labels`[cite: 4].
   - Normalized the pixel values by dividing by 255.0 to scale them between 0 and 1[cite: 4].
2. **Data Visualization:**
   - Visualized a 3x3 grid of 9 sample images from the training set, displaying them in grayscale with their respective class names (e.g., Ankle Boot, T-shirt/Top, Dress)[cite: 4].
   - Displayed a single isolated image (index 7, Pullover) to verify label mapping[cite: 4].
3. **Model Architecture:**
   - Initialized a `Sequential` neural network[cite: 4].
   - **Input Layer:** `Flatten` layer configured for an `input_shape` of (28, 28)[cite: 4].
   - **Hidden Layer:** `Dense` layer with 128 neurons and a `relu` activation function[cite: 4].
   - **Output Layer:** `Dense` layer with 10 neurons and a `softmax` activation function[cite: 4].
4. **Compilation & Training:**
   - Compiled the model using the `adam` optimizer and `sparse_categorical_crossentropy` loss function[cite: 4].
   - Tracked the `accuracy` metric[cite: 4].
   - Trained the model on the training data for 3 epochs[cite: 4].
5. **Evaluation:**
   - Evaluated the model on the test dataset, achieving a final test loss and test accuracy[cite: 4].

## Technologies Used
- `tensorflow` / `keras`[cite: 4]
- `matplotlib.pyplot` for visualization[cite: 4]