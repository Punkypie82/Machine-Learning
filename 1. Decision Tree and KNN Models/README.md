# Decision Tree and K-Nearest Neighbors Models

This directory contains implementations of two fundamental machine learning algorithms: Decision Trees and K-Nearest Neighbors (KNN). Both models are implemented from scratch to provide a deep understanding of the underlying algorithms.

## 📋 Contents

- **DecisionTree/**: Custom decision tree implementation with entropy-based splitting
- **KNN/**: K-Nearest Neighbors implementation with multiple distance metrics

## 🌳 Decision Tree Implementation

### Overview
The decision tree implementation uses the entropy-based information gain criterion for splitting nodes. The model includes advanced features like:

- **Custom Tree Structure**: Node-based implementation with support for both categorical and numerical features
- **Information Gain Calculation**: Uses entropy and Gini impurity for optimal splits
- **Pruning Support**: Maximum depth and minimum samples split parameters
- **Categorical Handling**: Advanced categorical splitting with multiple groups
- **Numerical Precision**: Configurable precision for numerical splits

### Key Features

#### Algorithm Components
- **TreeNode Class**: Represents individual nodes in the decision tree
- **DecisionTree Class**: Main implementation with training and prediction methods
- **Entropy Calculation**: Information theory-based splitting criterion
- **Best Split Selection**: Optimal feature and threshold selection

#### Advanced Capabilities
- **Mixed Data Types**: Handles both numerical and categorical features
- **Flexible Splitting**: Configurable numerical precision and maximum splits
- **Performance Metrics**: Comprehensive evaluation including accuracy, precision, recall, F1-score
- **Tree Visualization**: Text-based tree structure display

### Dataset: Loan Approval Prediction

The decision tree model is trained on a comprehensive loan approval dataset with the following features:

#### Features (13 total)
- **Personal Information**: Age, gender, education level
- **Financial Data**: Income, employment experience, home ownership
- **Loan Details**: Amount, interest rate, intent, percent of income
- **Credit History**: Credit score, credit history length, previous defaults

#### Target Variable
- **loan_status**: Binary classification (0 = Rejected, 1 = Approved)

#### Dataset Statistics
- **Size**: 45,000 samples
- **Features**: 13 input features
- **Classes**: Binary classification
- **Data Quality**: No missing values

### Performance Results

The custom implementation achieves competitive performance:
- **Accuracy**: ~91.7%
- **Precision**: ~87.3%
- **Recall**: ~73.2%
- **F1-Score**: ~79.6%

Performance is comparable to scikit-learn's DecisionTreeClassifier, validating the implementation quality.

## 🎯 K-Nearest Neighbors Implementation

### Overview
The KNN implementation provides a comprehensive approach to instance-based learning with support for various distance metrics and optimization techniques.

### Key Features

#### Distance Metrics
- **Euclidean Distance**: Standard L2 norm
- **Manhattan Distance**: L1 norm for high-dimensional data
- **Minkowski Distance**: Generalized distance metric
- **Custom Metrics**: Extensible framework for additional metrics

#### Optimization Features
- **Efficient Search**: Optimized neighbor finding algorithms
- **Cross-Validation**: Built-in k-value optimization
- **Weighted Voting**: Distance-based weight assignment
- **Scalability**: Memory-efficient implementation for large datasets

### Dataset: Adult Census Data

The KNN model uses the Adult Census dataset for income prediction:

#### Features
- **Demographics**: Age, education, marital status, race, gender
- **Employment**: Work class, occupation, hours per week
- **Geography**: Native country
- **Financial**: Capital gain/loss

#### Target Variable
- **Income Level**: Binary classification (<=50K, >50K)

#### Preprocessing
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Scaling**: Normalization for distance-based calculations
- **Missing Value Handling**: Appropriate imputation strategies

### Usage Examples

#### Decision Tree
```python
# Initialize and train the model
tree_classifier = DecisionTree(max_depth=7, numerical_precision=0.95, max_splits=7)
tree_classifier.fit(X_train, y_train)

# Make predictions
y_pred = tree_classifier.predict(X_test)

# Evaluate performance
metrics = tree_classifier.evaluate(y_test, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")

# Visualize tree structure
tree_classifier.print_tree()
```

#### K-Nearest Neighbors
```python
# Initialize KNN classifier
knn_classifier = KNNClassifier(k=5, distance_metric='euclidean')

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Evaluate performance
accuracy = knn_classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")
```

## 🔧 Implementation Details

### Decision Tree Algorithm
1. **Initialization**: Set hyperparameters (max_depth, min_samples_split)
2. **Split Generation**: Calculate possible splits for numerical and categorical features
3. **Information Gain**: Compute entropy-based information gain for each split
4. **Best Split Selection**: Choose split with maximum information gain
5. **Recursive Building**: Recursively build left and right subtrees
6. **Stopping Criteria**: Stop when max depth reached or pure nodes found

### KNN Algorithm
1. **Distance Calculation**: Compute distances to all training samples
2. **Neighbor Selection**: Find k nearest neighbors
3. **Voting**: Aggregate neighbor labels (with optional weighting)
4. **Prediction**: Return majority class or weighted average

## 📊 Comparative Analysis

Both implementations include comparisons with scikit-learn equivalents:

### Decision Tree vs Scikit-learn
- **Performance**: Comparable accuracy and metrics
- **Features**: Custom categorical handling vs built-in preprocessing
- **Flexibility**: More control over splitting criteria
- **Interpretability**: Enhanced tree visualization

### KNN vs Scikit-learn
- **Efficiency**: Optimized distance calculations
- **Metrics**: Multiple distance metric support
- **Scalability**: Memory-efficient for large datasets
- **Customization**: Extensible framework for custom metrics

## 🎓 Learning Objectives

After working through these implementations, you will understand:

1. **Tree Construction**: How decision trees split data recursively
2. **Information Theory**: Entropy and information gain concepts
3. **Instance-Based Learning**: How KNN makes predictions using similarity
4. **Distance Metrics**: Various ways to measure similarity between samples
5. **Hyperparameter Tuning**: Impact of k-value, max_depth, etc.
6. **Performance Evaluation**: Comprehensive metrics for classification tasks

## 📁 Files Description

### DecisionTree/
- `decision_tree.ipynb`: Complete implementation and analysis
- `loan_data.csv`: Loan approval dataset

### KNN/
- `knn.ipynb`: KNN implementation and experiments
- `adult.data`: Adult census dataset

## 🚀 Getting Started

1. **Navigate to the desired subdirectory**
2. **Open the Jupyter notebook**
3. **Run cells sequentially** to see the implementation
4. **Experiment with hyperparameters** to understand their impact
5. **Compare results** with scikit-learn implementations

## 📈 Extensions and Improvements

Potential enhancements for further learning:

### Decision Tree
- **Pruning Algorithms**: Post-pruning techniques
- **Ensemble Methods**: Random Forest implementation
- **Regression Trees**: Extend to continuous targets
- **Feature Importance**: Calculate and visualize feature importance

### KNN
- **Approximate Nearest Neighbors**: For large-scale datasets
- **Locality Sensitive Hashing**: Efficient similarity search
- **Adaptive k-values**: Dynamic k selection based on local density
- **Regression Extension**: KNN for continuous target variables

---

These implementations provide a solid foundation for understanding two fundamental machine learning algorithms while demonstrating best practices in algorithm implementation and evaluation.