# Deep Learning Assignment 3: Forward and Backpropagation Analysis

## Problem Statement
Implement forward propagation and backpropagation using TensorFlow/Keras[cite: 2]. Analyze the effect of different learning rates and the number of epochs on model performance[cite: 2].

## Dataset Overview
- **Name:** Fashion MNIST[cite: 2]
- **Structure:** Training and testing sets of 28x28 grayscale clothing images[cite: 2].
- **Classes:** 10 categories (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal)[cite: 2].

## Implementation Details
1. **Data Preprocessing:**
   - Visualized the first 10 images from the training set in a 2x5 grid[cite: 2].
   - Flattened the 28x28 images into 1D arrays of 784 features using `reshape(-1, 784)`[cite: 2].
   - Normalized pixel values by dividing by 255.0[cite: 2].
2. **Base Model Architecture:**
   - Defined a `create_model(lr)` function returning a Sequential network[cite: 2].
   - **Hidden Layers:** Two `Dense` layers with 128 and 64 neurons respectively, both using `relu` activation[cite: 2].
   - **Output Layer:** `Dense` layer with 10 neurons and `softmax` activation[cite: 2].
   - **Compilation:** Uses the `Adam` optimizer (with configurable learning rate), `sparse_categorical_crossentropy` loss, and tracks `accuracy`[cite: 2].
3. **Experiment 1: Learning Rate Analysis:**
   - Iterated through learning rates: `0.1`, `0.01`, and `0.001`[cite: 2].
   - Trained models for 5 epochs with a `batch_size=128`[cite: 2].
   - Recorded test loss and accuracy, storing them in a Pandas DataFrame[cite: 2].
   - Plotted a line graph of Learning Rate (log scale) vs. Test Accuracy to visualize performance degradation at high learning rates[cite: 2].
4. **Experiment 2: Epoch Analysis:**
   - Fixed the learning rate at `0.001`[cite: 2].
   - Iterated through epoch counts: `5`, `10`, and `20` (using a `validation_split=0.2`)[cite: 2].
   - Evaluated and plotted Number of Epochs vs. Test Accuracy[cite: 2].
5. **Final Performance Visualization:**
   - Trained a definitive model (`lr=0.001`) for 20 epochs with a 0.2 validation split[cite: 2].
   - Plotted Training vs. Validation Accuracy and Training vs. Validation Loss across the 20 epochs to identify potential overfitting[cite: 2].

## Technologies Used
- `tensorflow` / `keras`[cite: 2]
- `pandas`[cite: 2]
- `matplotlib.pyplot`[cite: 2]