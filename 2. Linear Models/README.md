# Linear Models

This directory contains implementations of fundamental linear machine learning algorithms: Linear Regression and Linear Classification. Both models demonstrate the mathematical foundations of linear methods and their practical applications.

## 📋 Contents

- **LinearRegression/**: From-scratch linear regression implementation with gradient descent
- **LinearClassification/**: Linear classification models including logistic regression

## 📈 Linear Regression Implementation

### Overview
A comprehensive implementation of linear regression using gradient descent optimization. The model includes both analytical and iterative solutions, providing deep insights into the mathematical foundations of linear modeling.

### Key Features

#### Algorithm Components
- **Gradient Descent**: Iterative optimization with configurable learning rates
- **Normal Equation**: Analytical solution for comparison
- **Regularization**: L1 (Lasso) and L2 (Ridge) regularization support
- **Feature Scaling**: Built-in normalization and standardization
- **Convergence Monitoring**: Loss tracking and early stopping

#### Advanced Capabilities
- **Multiple Variants**: Batch, stochastic, and mini-batch gradient descent
- **Learning Rate Scheduling**: Adaptive learning rate strategies
- **Cross-Validation**: Built-in k-fold cross-validation
- **Feature Engineering**: Polynomial features and interaction terms
- **Visualization**: Comprehensive plotting of results and convergence

### Dataset: Housing Price Prediction

The linear regression model uses housing market data for price prediction:

#### Features
- **Property Characteristics**: Size, number of rooms, age, condition
- **Location Factors**: Neighborhood, proximity to amenities
- **Market Indicators**: Historical prices, market trends
- **Economic Factors**: Interest rates, economic indicators

#### Target Variable
- **Price**: Continuous house price values

#### Preprocessing Steps
1. **Missing Value Handling**: Imputation strategies for incomplete data
2. **Feature Scaling**: Normalization for gradient descent convergence
3. **Outlier Detection**: Statistical methods for outlier identification
4. **Feature Selection**: Correlation analysis and feature importance

### Performance Metrics
- **Mean Squared Error (MSE)**: Primary loss function
- **Root Mean Squared Error (RMSE)**: Interpretable error metric
- **Mean Absolute Error (MAE)**: Robust error measurement
- **R-squared**: Coefficient of determination for model fit quality

## 🎯 Linear Classification Implementation

### Overview
Implementation of linear classification methods including logistic regression and linear discriminant analysis. The models demonstrate probabilistic classification and decision boundary learning.

### Key Features

#### Classification Algorithms
- **Logistic Regression**: Probabilistic binary and multiclass classification
- **Linear Discriminant Analysis**: Gaussian assumption-based classification
- **Perceptron**: Basic linear classifier with geometric interpretation
- **Support Vector Machine**: Linear SVM with margin maximization

#### Advanced Techniques
- **Regularization**: L1 and L2 penalties for overfitting prevention
- **Class Balancing**: Techniques for imbalanced datasets
- **Probability Calibration**: Reliable probability estimates
- **Feature Selection**: Automatic relevance determination

### Dataset: Customer Churn Prediction

The classification models use telecommunications customer data:

#### Features (20+ variables)
- **Customer Demographics**: Age, gender, location, tenure
- **Service Usage**: Call patterns, data usage, service types
- **Billing Information**: Monthly charges, total charges, payment methods
- **Service Quality**: Complaints, support interactions, satisfaction scores

#### Target Variable
- **Churn Status**: Binary classification (Stay = 0, Churn = 1)

#### Data Characteristics
- **Size**: Multiple datasets (train/validation/test splits)
- **Imbalance**: Realistic churn rates (~20-30%)
- **Mixed Types**: Numerical and categorical features
- **Real-world Complexity**: Missing values and noise

### Model Performance

#### Linear Regression Results
- **Training RMSE**: Optimized through hyperparameter tuning
- **Validation Performance**: Cross-validated results
- **Feature Importance**: Coefficient analysis and interpretation
- **Residual Analysis**: Model assumption validation

#### Classification Results
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Class-specific performance metrics
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Ranking and probability quality assessment

## 🔧 Implementation Details

### Linear Regression Algorithm
1. **Initialization**: Random weight initialization
2. **Forward Pass**: Linear combination of features
3. **Loss Calculation**: Mean squared error computation
4. **Gradient Computation**: Analytical gradient derivation
5. **Weight Update**: Gradient descent step
6. **Convergence Check**: Loss threshold or iteration limit

### Logistic Regression Algorithm
1. **Linear Combination**: Weighted sum of features
2. **Sigmoid Activation**: Probability transformation
3. **Log-Likelihood**: Probabilistic loss function
4. **Gradient Descent**: Iterative optimization
5. **Regularization**: Optional L1/L2 penalties
6. **Prediction**: Threshold-based classification

## 📊 Mathematical Foundations

### Linear Regression
```
Hypothesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
Cost Function: J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
Gradient: ∇J(θ) = (1/m) X^T(Xθ - y)
Update Rule: θ := θ - α∇J(θ)
```

### Logistic Regression
```
Hypothesis: h(x) = σ(θ^T x) = 1/(1 + e^(-θ^T x))
Cost Function: J(θ) = -(1/m) Σ[y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]
Gradient: ∇J(θ) = (1/m) X^T(σ(Xθ) - y)
```

## 💻 Usage Examples

### Linear Regression
```python
# Initialize and configure the model
lr_model = LinearRegression(learning_rate=0.01, max_iterations=1000)

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate performance
mse = lr_model.mean_squared_error(y_test, y_pred)
r2 = lr_model.r_squared(y_test, y_pred)

print(f"MSE: {mse:.3f}, R²: {r2:.3f}")

# Visualize results
lr_model.plot_convergence()
lr_model.plot_predictions(X_test, y_test, y_pred)
```

### Linear Classification
```python
# Initialize logistic regression
log_reg = LogisticRegression(learning_rate=0.01, regularization='l2', lambda_reg=0.01)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)

# Evaluate performance
accuracy = log_reg.accuracy(y_test, y_pred)
auc_score = log_reg.roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}")
```

## 📈 Visualization and Analysis

### Linear Regression Visualizations
- **Scatter Plots**: Feature vs target relationships
- **Residual Plots**: Model assumption validation
- **Learning Curves**: Training and validation performance
- **Feature Importance**: Coefficient magnitude analysis

### Classification Visualizations
- **Decision Boundaries**: 2D visualization of classification regions
- **ROC Curves**: True positive vs false positive rates
- **Confusion Matrix**: Detailed classification performance
- **Feature Coefficients**: Linear model interpretability

## 🎓 Learning Objectives

After completing these implementations, you will understand:

1. **Linear Algebra**: Matrix operations in machine learning
2. **Optimization**: Gradient descent and convergence properties
3. **Regularization**: Bias-variance tradeoff and overfitting prevention
4. **Probabilistic Models**: Maximum likelihood estimation
5. **Model Evaluation**: Comprehensive performance assessment
6. **Feature Engineering**: Data preprocessing and transformation

## 📁 Files Description

### LinearRegression/
- `linear_regression.ipynb`: Complete implementation and analysis
- `data/train.csv`: Training dataset
- `data/test.csv`: Testing dataset

### LinearClassification/
- `linear_classification.ipynb`: Classification models implementation
- `data/telco-customer-churn-train.csv`: Training data
- `data/telco-customer-churn-validation.csv`: Validation data
- `data/telco-customer-churn-test.csv`: Test data

## 🔍 Advanced Topics

### Regularization Techniques
- **Ridge Regression (L2)**: Coefficient shrinkage for stability
- **Lasso Regression (L1)**: Feature selection through sparsity
- **Elastic Net**: Combined L1 and L2 regularization
- **Cross-Validation**: Optimal regularization parameter selection

### Optimization Enhancements
- **Momentum**: Accelerated gradient descent
- **Adam Optimizer**: Adaptive learning rates
- **Learning Rate Scheduling**: Dynamic rate adjustment
- **Early Stopping**: Overfitting prevention

### Model Extensions
- **Polynomial Features**: Non-linear relationship modeling
- **Interaction Terms**: Feature combination effects
- **Multiclass Classification**: One-vs-rest and softmax approaches
- **Ensemble Methods**: Model combination strategies

## 🚀 Getting Started

1. **Choose a subdirectory** (LinearRegression or LinearClassification)
2. **Open the Jupyter notebook**
3. **Follow the step-by-step implementation**
4. **Experiment with hyperparameters**
5. **Compare with scikit-learn implementations**
6. **Analyze results and visualizations**

## 📊 Comparative Analysis

### Performance Comparison
Both implementations include detailed comparisons with established libraries:
- **Scikit-learn**: Baseline performance comparison
- **Convergence Speed**: Optimization efficiency analysis
- **Memory Usage**: Computational resource requirements
- **Scalability**: Performance on different dataset sizes

### Implementation Benefits
- **Educational Value**: Clear mathematical derivations
- **Customization**: Easy modification for specific needs
- **Transparency**: Full control over algorithm behavior
- **Understanding**: Deep insights into model mechanics

---

These linear model implementations provide a solid foundation for understanding the mathematical principles underlying many machine learning algorithms while demonstrating practical applications in regression and classification tasks.