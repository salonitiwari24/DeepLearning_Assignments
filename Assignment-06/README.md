# Deep Learning Assignment 6: Convolutional Neural Network (CNN)

## Problem Statement
Design and implement a Convolutional Neural Network (CNN) for image classification using the Rice Leaf Disease dataset[cite: 1].

## Dataset Overview
- **Name:** Rice Leaf Diseases[cite: 1]
- **Structure:** 120 total images[cite: 1].
- **Classes:** 3 categories (Bacterial leaf blight, Brown spot, Leaf smut)[cite: 1].

## Implementation Details
1. **Data Organization & Splitting:**
   - Created a Pandas DataFrame mapping each image's file path to its class label[cite: 1].
   - Visualized class distribution using a Seaborn countplot and displayed a sample image from each class[cite: 1].
   - Split the data into Training (70%), Validation (15%), and Testing (15%) subsets using a stratified `train_test_split`[cite: 1].
2. **Preprocessing & Data Augmentation:**
   - Resized all images to a standard `(128, 128)` resolution[cite: 1].
   - Implemented an `ImageDataGenerator` for the training set featuring data augmentation: rescaling (1./255), rotation range (25), width/height shift (0.15), zoom range (0.20), shear range (0.15), and horizontal flipping[cite: 1].
   - Configured `ImageDataGenerator` for validation and test sets using only rescaling[cite: 1].
   - Created `flow_from_dataframe` generators with a batch size of 16[cite: 1].
3. **CNN Architecture & Hyperparameter Tuning:**
   - Defined a dynamic `build_model` function for `Keras Tuner RandomSearch`[cite: 1].
   - **Tuning Space:** 
     - 2 to 3 Conv2D blocks (searching for 32, 64, or 128 filters per block)[cite: 1].
     - Included `MaxPooling2D` layers (pool_size=2) after convolutions[cite: 1].
     - 1 Flatten layer feeding into a Dense layer (searching for 64, 128, or 256 units)[cite: 1].
     - 1 Dropout layer (searching rates between 0.2 and 0.5)[cite: 1].
     - Output Dense layer (3 units, `softmax`)[cite: 1].
     - Adam optimizer learning rates (0.001, 0.0005, 0.0001)[cite: 1].
   - Ran `tuner.search` to find the optimal architecture targeting `val_accuracy`[cite: 1].
4. **Training & Evaluation:**
   - Extracted the best hyperparameters and model structure[cite: 1].
   - Implemented an `EarlyStopping` callback (monitoring `val_loss`, patience=7, restoring best weights)[cite: 1].
   - Trained the optimized model for up to 30 epochs[cite: 1].
   - Plotted side-by-side graphs for Training vs. Validation Accuracy and Training vs. Validation Loss[cite: 1].
   - Generated final predictions using `np.argmax` and evaluated performance via accuracy score, a detailed `classification_report`, and a Seaborn heatmap `confusion_matrix`[cite: 1].

## Technologies Used
- `tensorflow` / `keras` (Layers, ImageDataGenerator, Callbacks)[cite: 1]
- `keras_tuner`[cite: 1]
- `PIL` (Image processing)[cite: 1]
- `sklearn` (Metrics, train_test_split)[cite: 1]
- `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`[cite: 1]