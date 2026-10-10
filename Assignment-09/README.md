
# Deep Learning Assignment: Image Classification Using Pre-trained ViT and CNN

## Problem Statement

Implement an image classification system using a Convolutional Neural Network (CNN) and a pre-trained Vision Transformer (ViT) on the CIFAR-10 dataset. Train the CNN from scratch and fine-tune the pre-trained ViT model for ten-class image classification. Evaluate both models using accuracy, precision, recall, and F1-score, and compare their performance.

## Dataset Overview

- **Name:** CIFAR-10 (Canadian Institute for Advanced Research)
- **Classes:** 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **Total Images:** 60,000 color images
- **Image Size:** 32 × 32 pixels with 3 RGB channels
- **CNN Training Data:** 50,000 images, with 10% used for validation
- **CNN Testing Data:** 10,000 images
- **ViT Training Data:** 5,000 images
- **ViT Testing Data:** 1,000 images
- **ViT Input Size:** 224 × 224 pixels

## Implementation Details

1. **Dataset Loading and Exploration**
   - Loaded CIFAR-10 using TensorFlow/Keras.
   - Normalized image pixel values to the range `[0, 1]`.
   - Defined the ten class labels and visualized sample images.

2. **CNN Model Development**
   - Created a sequential CNN with two convolutional layers containing 32 and 64 filters.
   - Applied max-pooling layers to reduce spatial dimensions.
   - Added a flattening layer, a dense layer with 128 neurons, and a softmax output layer for ten-class classification.

3. **CNN Training**
   - Trained the CNN for five epochs using the Adam optimizer.
   - Used sparse categorical cross-entropy as the loss function.
   - Used a batch size of 64 and 10% of the training data for validation.

4. **CNN Evaluation**
   - Evaluated the CNN on all 10,000 test images.
   - Generated predictions and calculated accuracy, precision, recall, and F1-score.
   - Displayed a classification report for all ten classes.

5. **Pre-trained ViT Model Loading**
   - Loaded `google/vit-base-patch16-224` using Hugging Face Transformers.
   - Used `ViTImageProcessor` to resize and preprocess CIFAR-10 images.
   - Replaced the original classification head with a ten-class output layer.

6. **ViT Data Preprocessing**
   - Selected 5,000 training images and 1,000 testing images.
   - Resized images to 224 × 224 pixels.
   - Processed images in small batches to reduce RAM consumption.

7. **ViT Fine-tuning and Evaluation**
   - Fine-tuned the pre-trained ViT for two epochs using the Adam optimizer.
   - Used sparse categorical cross-entropy as the loss function.
   - Evaluated predictions on the 1,000-image test subset.
   - Generated a classification report containing precision, recall, and F1-score.

8. **Performance Comparison**
   - Compared CNN and ViT using accuracy, weighted precision, weighted recall, and weighted F1-score.
   - Evaluated both models on the same first 1,000 test images for the comparison.
   - Plotted a bar chart to visualize the accuracy difference.

## Technologies Used

- `TensorFlow / Keras` — CNN development and dataset handling
- `tf-keras` — Compatibility with the TensorFlow ViT model
- `Hugging Face Transformers` — Pre-trained Vision Transformer
- `scikit-learn` — Classification metrics and evaluation
- `NumPy` — Numerical processing
- `Pandas` — Results comparison table
- `Matplotlib` — Image visualization and accuracy plot
- `Google Colab` — Implementation and execution

## Results

| Metric                       | CNN    | ViT    |
|------------------------------|--------|--------|
| Accuracy (comparison subset) | 68.20% | 95.50% |
| Weighted Precision           | 68.97% | 95.83% |
| Weighted Recall              | 68.20% | 95.50% |
| Weighted F1-score            | 67.85% | 95.50% |

- **CNN Accuracy on the Full Test Set:** 67.22%
- **ViT Accuracy on the Selected Test Subset:** 95.50%

*Note: The comparison uses the CNN predictions on the first 1,000 test images and the ViT predictions on the same subset. The CNN's full test-set accuracy was 67.22%. The models were trained on different amounts of data, so the results should be interpreted with this limitation in mind.*

## Conclusion

This assignment demonstrates image classification using a CNN trained from scratch and a pre-trained Vision Transformer fine-tuned on CIFAR-10. The CNN achieved 67.22% accuracy on the complete test set, while the ViT achieved 95.50% accuracy on its selected test subset. The results illustrate the potential benefits of transfer learning with pre-trained transformer models for image classification. However, the models were trained using different amounts of data. Using identical training and testing subsets would provide a fairer comparison.
