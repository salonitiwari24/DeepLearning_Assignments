# Deep Learning Assignment 2: Multilayer Perceptron (MLP) Implementation

## Problem Statement
Design and implement a Multilayer Perceptron (MLP) for classification of the Iris dataset, and evaluate its performance using accuracy and a confusion matrix[cite: 3].

## Dataset Overview
- **Name:** Iris Dataset (imported via `sklearn.datasets`)[cite: 3].
- **Features:** 4 numeric inputs (sepal length, sepal width, petal length, petal width)[cite: 3].
- **Classes:** 3 target species (setosa, versicolor, virginica)[cite: 3].

## Implementation Details
1. **Data Exploration & Preprocessing:**
   - Loaded the dataset into a Pandas DataFrame and appended the target labels as a "Species" column[cite: 3].
   - Printed data info, statistical descriptions (`describe()`), and verified class distribution (`value_counts()`)[cite: 3].
   - Checked for missing values using `isnull().sum()` (0 missing values found)[cite: 3].
   - Split the data into training (80%) and testing (20%) subsets using `train_test_split` with `random_state=42`[cite: 3].
2. **Model Architecture & Training:**
   - Initialized an `MLPClassifier` from Scikit-Learn[cite: 3].
   - **Hyperparameters:** Configured with two hidden layers (`hidden_layer_sizes=(10,10)`), `relu` activation, `adam` solver, and `max_iter=1000`[cite: 3].
   - Trained the model on `X_train` and `y_train`[cite: 3].
3. **Evaluation Metrics:**
   - Generated predictions (`y_pred`) on the test set (`X_test`)[cite: 3].
   - Calculated the overall accuracy score (achieved 0.9333)[cite: 3].
   - Computed a Confusion Matrix and visualized it using a Seaborn `heatmap` (annotated with integers, using the 'Blues' colormap)[cite: 3].
   - Generated a comprehensive classification report detailing precision, recall, f1-score, and support for each species class[cite: 3].
   - Displayed a DataFrame directly comparing Actual vs. Predicted values for the first 15 test samples[cite: 3].

## Technologies Used
- `sklearn` (`MLPClassifier`, `train_test_split`, `load_iris`, metrics)[cite: 3]
- `pandas` & `numpy`[cite: 3]
- `matplotlib.pyplot` & `seaborn`[cite: 3]