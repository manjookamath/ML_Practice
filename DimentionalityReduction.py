# Linear Regression with sklearn

# Load the dataset from the working directory

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
x_values = np.load('x_values-1.npy')
y_values = np.load('y_values-1.npy')

# Visualize the data with a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.7, color='blue', label='Observed Data')
plt.xlabel('X values (Regular Variable)')
plt.ylabel('Y values (Target Variable)')
plt.title('Scatter Plot: X values vs Y values')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Print some basic statistics
print(f"X values shape: {x_values.shape}")
print(f"Y values shape: {y_values.shape}")
print(f"X values range: {x_values.min():.2f} to {x_values.max():.2f}")
print(f"Y values range: {y_values.min():.2f} to {y_values.max():.2f}")

# Reshape x_values if it's 1D (required for scikit-learn)
if len(x_values.shape) == 1:
    x_values = x_values.reshape(-1, 1)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(x_values, y_values)

# Print model coefficients
print(f"\nModel Coefficients:")
print(f"Slope (coefficient): {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Make predictions on the training data for visualization
y_pred = model.predict(x_values)

# Calculate R-squared score
r2 = r2_score(y_values, y_pred)
print(f"R-squared score: {r2:.4f}")

# Visualize the regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, alpha=0.7, color='blue', label='Observed Data')
plt.plot(x_values, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('X values (Regular Variable)')
plt.ylabel('Y values (Target Variable)')
plt.title('Linear Regression: X values vs Y values')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Predict for an unseen value
unseen_x = np.array([[x_values.max() + 1]])
predicted_y = model.predict(unseen_x)

print(f"\nPrediction for unseen X value {unseen_x[0][0]:.2f}:")
print(f"Predicted Y value: {predicted_y[0][0]:.4f}")

# You can also predict for multiple unseen values
multiple_unseen = np.array([[2.5], [3.0], [4.2]])
multiple_predictions = model.predict(multiple_unseen)

print(f"\nPredictions for multiple unseen X values:")
for x, y_pred in zip(multiple_unseen.flatten(), multiple_predictions):
    print(f"X = {x:.2f} -> Predicted Y = {y_pred[0]:.4f}")

# Predict and print the y value for z = 0.48
# Note: The input must be a 2D array (matrix) with shape (n_samples, n_features)
z_value = 0.48
z_matrix = np.array([[z_value]])  # Convert scalar to 2D array

predicted_y = model.predict(z_matrix)

print(f"\nPrediction for z = {z_value}:")
print(f"Predicted y value: {predicted_y[0][0]:.4f}")

#Principle Component Analysis with sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
wine_data = np.load('wineData-1.npy')
wine_labels = np.load('wineLabels-1.npy')

print("Wine Data shape:", wine_data.shape)
print("Wine Labels shape:", wine_labels.shape)
print("\nFirst 5 data points:")
print(wine_data[:5])
print("\nFirst 5 labels:")
print(wine_labels[:5])
print("\nUnique labels:", np.unique(wine_labels))

# Split the data into training and testing sets (80% train, 20% test)
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(len(wine_data))
split_idx = int(0.8 * len(wine_data))

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

train_data = wine_data[train_idx]
train_labels = wine_labels[train_idx]
test_data = wine_data[test_idx]
test_labels = wine_labels[test_idx]

print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing data shape: {test_data.shape}")
print(f"Testing labels shape: {test_labels.shape}")

# Visualize the original data with train/test split
plt.figure(figsize=(15, 5))

# Plot 1: Original feature space (Feature 0 vs Feature 1)
plt.subplot(1, 2, 1)
# Training data
for label in np.unique(train_labels):
    mask = train_labels == label
    plt.scatter(train_data[mask, 0], train_data[mask, 1],
                marker='o', label=f'Class {label} (Train)', alpha=0.7)
# Testing data
for label in np.unique(test_labels):
    mask = test_labels == label
    plt.scatter(test_data[mask, 0], test_data[mask, 1],
                marker='x', s=100, label=f'Class {label} (Test)', alpha=0.7)

plt.xlabel('Feature 0 (Alcohol)')
plt.ylabel('Feature 1 (Malic Acid)')
plt.title('Original Data: Train/Test Split\n(Feature 0 vs Feature 1)')
plt.legend()
plt.grid(True, alpha=0.3)

# Standardize the data before PCA (important for PCA)
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Initialize and fit PCA
pca = PCA(n_components=2)
pca.fit(train_data_scaled)

# Apply PCA transformation
train_data_pca = pca.transform(train_data_scaled)
test_data_pca = pca.transform(test_data_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")

# Plot 2: PCA transformed data
plt.subplot(1, 2, 2)
# Training data in PCA space
for label in np.unique(train_labels):
    mask = train_labels == label
    plt.scatter(train_data_pca[mask, 0], train_data_pca[mask, 1],
                marker='o', label=f'Class {label} (Train)', alpha=0.7)
# Testing data in PCA space
for label in np.unique(test_labels):
    mask = test_labels == label
    plt.scatter(test_data_pca[mask, 0], test_data_pca[mask, 1],
                marker='x', s=100, label=f'Class {label} (Test)', alpha=0.7)

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Projection: 2D Visualization\n(13D → 2D)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Just the PCA space for clarity
plt.figure(figsize=(10, 8))

# Training data in PCA space
for label in np.unique(train_labels):
    mask = train_labels == label
    plt.scatter(train_data_pca[mask, 0], train_data_pca[mask, 1],
                marker='o', s=80, label=f'Class {label} (Train)', alpha=0.8)

# Testing data in PCA space
for label in np.unique(test_labels):
    mask = test_labels == label
    plt.scatter(test_data_pca[mask, 0], test_data_pca[mask, 1],
                marker='X', s=150, linewidth=2, label=f'Class {label} (Test)', alpha=0.8)

plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Wine Dataset: PCA Dimensionality Reduction (13D → 2D)\nTraining (circles) vs Testing (crosses)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some information about the PCA transformation
plt.figtext(0.02, 0.02, f'Total explained variance: {np.sum(pca.explained_variance_ratio_):.1%}',
            fontsize=10, style='italic')

plt.show()

#Linear Discriminant  Analysis with sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Using the same train:test split as in task 3.2 (assuming 80:20 split with random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create LDA object with 2 components
lda = LinearDiscriminantAnalysis(n_components=2)

# Fit the model to training data and training labels
# We need labels because LDA is supervised - it finds directions that maximize class separation
lda.fit(X_train, y_train)

# Apply dimensionality reduction transform to training data
X_train_lda = lda.transform(X_train)

# Apply dimensionality reduction transform to testing data
X_test_lda = lda.transform(X_test)

# Visualize the reduced-dimensionality data
plt.figure(figsize=(12, 5))

# Plot training data
plt.subplot(1, 2, 1)
for i, color, marker in zip([0, 1, 2], ['red', 'blue', 'green'], ['o', 's', '^']):
    plt.scatter(X_train_lda[y_train == i, 0],
                X_train_lda[y_train == i, 1],
                c=color, marker=marker,
                label=f'Class {i}', alpha=0.7, edgecolors='w', s=50)
plt.title('LDA - Training Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot testing data
plt.subplot(1, 2, 2)
for i, color, marker in zip([0, 1, 2], ['red', 'blue', 'green'], ['x', '+', 'd']):
    plt.scatter(X_test_lda[y_test == i, 0],
                X_test_lda[y_test == i, 1],
                c=color, marker=marker,
                label=f'Class {i}', alpha=0.7, s=60, linewidth=2)
plt.title('LDA - Testing Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Alternative: Combined visualization
plt.figure(figsize=(10, 8))

# Plot training data (filled markers)
for i, color in zip([0, 1, 2], ['red', 'blue', 'green']):
    plt.scatter(X_train_lda[y_train == i, 0],
                X_train_lda[y_train == i, 1],
                c=color, marker='o',
                label=f'Train Class {i}', alpha=0.7, s=50, edgecolors='w')

# Plot testing data (hollow markers)
for i, color in zip([0, 1, 2], ['red', 'blue', 'green']):
    plt.scatter(X_test_lda[y_test == i, 0],
                X_test_lda[y_test == i, 1],
                c=color, marker='s', facecolors='none',
                label=f'Test Class {i}', alpha=0.7, s=60, linewidth=2)

plt.title('LDA - Combined Training and Testing Data')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print some information about the transformation
print(f"Original feature space: {X_train.shape[1]} dimensions")
print(f"Reduced feature space: {X_train_lda.shape[1]} dimensions")
print(f"Training set size: {X_train_lda.shape[0]} samples")
print(f"Testing set size: {X_test_lda.shape[0]} samples")

#Application of Skill to derive minimum components which explain max variance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data and labels
data = np.load('task3_5_data.npy')
labels = np.load('task3_5_labels.npy')

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Number of unique labels: {np.unique(labels)}")

# Perform PCA without dimensionality reduction first to analyze variance
pca_full = PCA()
pca_full.fit(data)

# Calculate cumulative explained variance ratio
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find the minimum number of components to retain at least 90% variance
min_components = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\nMinimum number of components to retain 90% variance: {min_components}")
print(f"Explained variance with {min_components} components: {cumulative_variance[min_components-1]:.4f}")

# Let's also check the variance for a range of components around this value
print("\nVariance retention for nearby component counts:")
for n in range(max(1, min_components-2), min(min_components+3, len(cumulative_variance)+1)):
    var_exp = cumulative_variance[n-1]
    print(f"Components: {n:2d} -> Variance: {var_exp:.4f} ({var_exp*100:.2f}%)")

# Visualize the explained variance
plt.figure(figsize=(12, 5))

# Plot 1: Scree plot
plt.subplot(1, 2, 1)
components = range(1, len(pca_full.explained_variance_ratio_) + 1)
plt.plot(components, pca_full.explained_variance_ratio_, 'bo-', linewidth=2, markersize=4, label='Individual Variance')
plt.plot(components, cumulative_variance, 'ro-', linewidth=2, markersize=4, label='Cumulative Variance')
plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Variance Threshold')
plt.axvline(x=min_components, color='green', linestyle='--', alpha=0.7)
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - PCA Variance Analysis')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Focus on cumulative variance with threshold
plt.subplot(1, 2, 2)
plt.plot(components, cumulative_variance, 'ro-', linewidth=2, markersize=4)
plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='90% Variance Threshold')
plt.axvline(x=min_components, color='green', linestyle='--', alpha=0.7, label=f'Min Components: {min_components}')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title(f'Cumulative Variance (90% at {min_components} components)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Now let's visualize the data in 2D and 3D using the actual PCA transformation
# First with the minimum components for 90% variance
pca_optimal = PCA(n_components=min_components)
data_optimal = pca_optimal.fit_transform(data)

print(f"\nOptimal PCA transformation shape: {data_optimal.shape}")

# Also create 2D and 3D versions for visualization
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(data)

pca_3d = PCA(n_components=3)
data_3d = pca_3d.fit_transform(data)

print(f"Variance explained by 2 components: {np.sum(pca_2d.explained_variance_ratio_):.4f}")
print(f"Variance explained by 3 components: {np.sum(pca_3d.explained_variance_ratio_):.4f}")

# Visualize the data in 2D with labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
plt.colorbar(scatter, label='Class Labels')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'2D PCA Projection (Total: {np.sum(pca_2d.explained_variance_ratio_):.2%} variance)')
plt.grid(True, alpha=0.3)
plt.show()

# If we have 3D capability, let's also show that
try:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                        c=labels, cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Class Labels')
    ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
    ax.set_title(f'3D PCA Projection (Total: {np.sum(pca_3d.explained_variance_ratio_):.2%} variance)')
    plt.show()
except ImportError:
    print("3D plotting not available")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original data dimensions: {data.shape[1]}")
print(f"Minimum components for 90% variance: {min_components}")
print(f"Variance retained: {cumulative_variance[min_components-1]:.4f}")
print(f"Reduction in dimensionality: {data.shape[1]} -> {min_components}")
print(f"Compression ratio: {min_components/data.shape[1]:.2%}")

# Task 3.4 - Principal Component Analysis by hand

import numpy as np
import matplotlib.pyplot as plt

def pca_by_hand(X_train, X_test, y_train, y_test, n_components=2):
    """
    Implement PCA using SVD

    Parameters:
    X_train: training data (samples x features)
    X_test: testing data (samples x features)
    y_train: training labels
    y_test: testing labels
    n_components: number of principal components to keep
    """

    # Step 1: Mean-centre the training data
    train_mean = np.mean(X_train, axis=0)
    X_train_centered = X_train - train_mean

    # Step 2: Calculate Singular Value Decomposition
    # Note: numpy's svd returns U, S, Vt where Vt = V^T
    U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)

    # Step 3: Project training data into 2D PCA space
    # We take the first n_components rows of Vt (which are the principal components)
    projection_matrix = Vt[:n_components, :]  # Shape: (n_components, n_features)

    # Why the transpose?
    # projection_matrix has shape (n_components, n_features)
    # X_train_centered has shape (n_samples, n_features)
    # To project: X_train_centered @ projection_matrix.T
    # This gives us (n_samples, n_components)
    projected_train = X_train_centered @ projection_matrix.T

    # Step 4: Project testing data
    # Mean-centre test data using training mean (to avoid data leakage)
    X_test_centered = X_test - train_mean

    # Use same projection matrix (to ensure consistent transformation)
    projected_test = X_test_centered @ projection_matrix.T

    return projected_train, projected_test, projection_matrix, train_mean

def visualize_pca(projected_train, projected_test, y_train, y_test):
    """
    Visualize the projected training and testing data
    """
    plt.figure(figsize=(12, 5))

    # Plot training data
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(projected_train[:, 0], projected_train[:, 1],
                         c=y_train, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Training Data in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plot testing data
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(projected_test[:, 0], projected_test[:, 1],
                         c=y_test, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Testing Data in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()

# Example usage with your data from Task 3.2
# Assuming you have X_train, X_test, y_train, y_test from your previous task

# Perform PCA
projected_train, projected_test, projection_matrix, train_mean = pca_by_hand(
    X_train, X_test, y_train, y_test, n_components=2
)

# Visualize results
visualize_pca(projected_train, projected_test, y_train, y_test)

# Print some information about the transformation
print(f"Original training data shape: {X_train.shape}")
print(f"Projected training data shape: {projected_train.shape}")
print(f"Projection matrix shape: {projection_matrix.shape}")
print(f"Dimensionality reduced from {X_train.shape[1]} to {projected_train.shape[1]}")

def explained_variance(X_train):
    """Calculate explained variance ratio"""
    train_mean = np.mean(X_train, axis=0)
    X_centered = X_train - train_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Explained variance ratio
    explained_variance = (S ** 2) / (X_centered.shape[0] - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(explained_variance_ratio) + 1),
             np.cumsum(explained_variance_ratio), 'bo-')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()

    return explained_variance_ratio

# Use this to see how much variance is captured by 2 components
variance_ratio = explained_variance(X_train)
print(f"Variance explained by first 2 components: {np.sum(variance_ratio[:2]):.3f}")

"""1. Why do we not compute a projection matrix and mean value for the testing sets?

**We use the same transformation learned from the training data for several important reasons:**

- **Prevention of Data Leakage**: If we computed statistics (mean, variance, etc.) from the test set, we'd be using information from the test set to build our transformation, which contaminates the evaluation.

- **Real-world Simulation**: In practice, we deploy models on completely new, unseen data. We must use the same transformation that was learned during training.

- **Consistent Feature Space**: The training and test data need to be in the same feature space. If we computed different means/transformations, they wouldn't be comparable.

- **Reproducibility**: The transformation should be deterministic based on the training data.

```python
# Correct approach - use training statistics for both
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std  # Use training stats!
```

## 2. Why does LDA give us nice distinct clusters for our Wine Dataset when PCA does not?

**LDA is supervised while PCA is unsupervised:**

| **LDA (Linear Discriminant Analysis)** | **PCA (Principal Component Analysis)** |
|----------------------------------------|----------------------------------------|
| **Supervised** - uses class labels | **Unsupervised** - ignores class labels |
| Maximizes **between-class variance** | Maximizes **total variance** |
| Minimizes **within-class variance** | No consideration of class separation |
| Finds directions that **separate classes** | Finds directions of **maximum spread** |

**Mathematical difference:**
- **PCA**: Maximizes variance: `argmax(wᵀΣw)` where Σ is total covariance matrix
- **LDA**: Maximizes ratio: `argmax(wᵀS_b w / wᵀS_w w)` where S_b is between-class and S_w is within-class scatter

For the Wine Dataset, the classes are well-separated in the original feature space, so LDA can find projections that explicitly separate them, while PCA might prioritize directions with high variance that don't necessarily align with class boundaries.

## 3. Benefits and Drawbacks of Dimensionality Reduction

### **Benefits:**
- **Reduced Computational Complexity**: Fewer dimensions = faster training and inference
- **Prevents Overfitting**: Removes noise and redundant features (curse of dimensionality)
- **Improved Visualization**: Can plot high-dimensional data in 2D/3D
- **Feature Extraction**: Creates new, more meaningful features
- **Memory Efficiency**: Smaller storage requirements
- **Noise Reduction**: Removes irrelevant variations in data

### **Drawbacks:**
- **Information Loss**: Some variance/discriminative power is always lost
- **Interpretability**: New features may not have clear physical meaning
- **Computational Overhead**: Need to compute transformation
- **Parameter Tuning**: Choosing the right number of components can be challenging
- **Linearity Assumption**: Methods like PCA/LDA assume linear relationships

## 4. Using LDA for Class Prediction

LDA can be used directly for classification by leveraging its probabilistic framework:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix

# LDA for classification (automatically reduces dimensions)
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = lda_classifier.predict(X_test)
y_prob = lda_classifier.predict_proba(X_test)

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# You can also get the transformed features for visualization
X_lda = lda_classifier.transform(X_test)
```

**How LDA prediction works:**
1. **Bayes' Theorem**: LDA uses Bayes' rule to compute posterior probabilities
2. **Class Conditional Distributions**: Assumes each class has Gaussian distribution with same covariance
3. **Decision Boundary**: Classifies based on which class has highest posterior probability
4. **Linear Decision Boundaries**: Results in linear separation between classes

**The prediction rule:**
```
Assign x to class k that maximizes: P(class=k | x) ∝ P(x | class=k) × P(class=k)
```

This makes LDA both a dimensionality reduction technique AND a classifier, which is particularly useful when you want interpretable features along with classification capability.
"""
