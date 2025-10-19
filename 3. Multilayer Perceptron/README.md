# Multilayer Perceptron (Neural Network)

This directory contains a comprehensive implementation of a Multilayer Perceptron (MLP) neural network built from scratch. The implementation demonstrates the fundamental concepts of deep learning including forward propagation, backpropagation, and gradient descent optimization.

## 📋 Overview

The Multilayer Perceptron is a feedforward neural network that serves as the foundation for understanding deep learning. This implementation provides:

- **From-scratch neural network** using NumPy and PyTorch
- **Comprehensive backpropagation** algorithm implementation
- **Multiple activation functions** and optimization techniques
- **Detailed mathematical explanations** of neural network concepts
- **Performance comparison** with established frameworks

## 🧠 Neural Network Architecture

### Network Structure
- **Input Layer**: Configurable number of input neurons
- **Hidden Layers**: Multiple hidden layers with customizable sizes
- **Output Layer**: Appropriate for classification or regression tasks
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax support
- **Dropout**: Regularization technique for overfitting prevention

### Key Components

#### Forward Propagation
```python
# Layer computation
z = W @ a_prev + b
a = activation_function(z)
```

#### Backpropagation
```python
# Gradient computation
dW = (1/m) * dZ @ A_prev.T
db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
dA_prev = W.T @ dZ
```

#### Weight Updates
```python
# Gradient descent
W = W - learning_rate * dW
b = b - learning_rate * db
```

## 🔧 Implementation Features

### Core Architecture
- **Modular Design**: Separate classes for layers, activations, and optimizers
- **Flexible Configuration**: Easy modification of network architecture
- **Memory Efficient**: Optimized matrix operations for large datasets
- **Extensible Framework**: Easy addition of new components

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Most common for hidden layers
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Binary classification output
- **Tanh**: `f(x) = (e^x - e^(-x))/(e^x + e^(-x))` - Centered activation
- **Softmax**: `f(x_i) = e^(x_i)/Σe^(x_j)` - Multiclass classification
- **Leaky ReLU**: `f(x) = max(αx, x)` - Addresses dying ReLU problem

### Loss Functions
- **Mean Squared Error**: For regression tasks
- **Cross-Entropy**: For classification tasks
- **Binary Cross-Entropy**: For binary classification
- **Categorical Cross-Entropy**: For multiclass classification

### Optimization Algorithms
- **Gradient Descent**: Basic optimization algorithm
- **Momentum**: Accelerated gradient descent
- **Adam**: Adaptive moment estimation
- **RMSprop**: Root mean square propagation
- **Learning Rate Scheduling**: Adaptive learning rate strategies

### Regularization Techniques
- **L1 Regularization**: Sparsity-inducing penalty
- **L2 Regularization**: Weight decay for smoother models
- **Dropout**: Random neuron deactivation during training
- **Early Stopping**: Prevent overfitting through validation monitoring
- **Batch Normalization**: Normalize layer inputs for stable training

## 📊 Mathematical Foundations

### Forward Propagation Mathematics
For layer `l`:
```
Z^[l] = W^[l] * A^[l-1] + b^[l]
A^[l] = g^[l](Z^[l])
```

Where:
- `W^[l]`: Weight matrix for layer l
- `b^[l]`: Bias vector for layer l
- `g^[l]`: Activation function for layer l
- `A^[l]`: Activations for layer l

### Backpropagation Mathematics
For layer `l`:
```
dZ^[l] = dA^[l] * g'^[l](Z^[l])
dW^[l] = (1/m) * dZ^[l] * A^[l-1].T
db^[l] = (1/m) * sum(dZ^[l], axis=1, keepdims=True)
dA^[l-1] = W^[l].T * dZ^[l]
```

### Cost Function
For classification:
```
J = -(1/m) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + λ*Σ(W²)
```

## 💻 Usage Examples

### Basic Neural Network
```python
# Initialize the neural network
nn = NeuralNetwork(
    layers=[784, 128, 64, 10],  # Input, hidden, output sizes
    activations=['relu', 'relu', 'softmax'],
    learning_rate=0.001,
    regularization='l2',
    lambda_reg=0.01
)

# Train the network
history = nn.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=True
)

# Make predictions
predictions = nn.predict(X_test)
probabilities = nn.predict_proba(X_test)

# Evaluate performance
accuracy = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.3f}")
```

### Advanced Configuration
```python
# Custom neural network with advanced features
nn = NeuralNetwork(
    layers=[784, 256, 128, 64, 10],
    activations=['relu', 'relu', 'relu', 'softmax'],
    optimizer='adam',
    learning_rate=0.001,
    beta1=0.9,  # Adam parameter
    beta2=0.999,  # Adam parameter
    dropout_rate=0.2,
    batch_norm=True,
    early_stopping=True,
    patience=10
)

# Train with learning rate scheduling
scheduler = LearningRateScheduler(
    initial_lr=0.001,
    decay_rate=0.95,
    decay_steps=1000
)

history = nn.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    scheduler=scheduler,
    save_best=True
)
```

## 📈 Training Process

### Data Preprocessing
1. **Normalization**: Scale input features to [0,1] or standardize
2. **One-Hot Encoding**: Convert categorical labels to binary vectors
3. **Train-Validation Split**: Separate data for model validation
4. **Batch Creation**: Organize data into mini-batches for training

### Training Loop
1. **Forward Pass**: Compute predictions through the network
2. **Loss Calculation**: Measure prediction error
3. **Backward Pass**: Compute gradients via backpropagation
4. **Parameter Update**: Apply gradients to weights and biases
5. **Validation**: Evaluate performance on validation set
6. **Monitoring**: Track metrics and implement early stopping

### Hyperparameter Tuning
- **Learning Rate**: Critical for convergence speed and stability
- **Network Architecture**: Number of layers and neurons per layer
- **Batch Size**: Trade-off between speed and gradient quality
- **Regularization**: Balance between underfitting and overfitting
- **Activation Functions**: Choice impacts gradient flow and expressiveness

## 📊 Performance Analysis

### Training Metrics
- **Training Loss**: Monitor convergence and overfitting
- **Validation Loss**: Early stopping and generalization assessment
- **Training Accuracy**: Learning progress on training data
- **Validation Accuracy**: True performance indicator

### Visualization Tools
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Performance improvement visualization
- **Weight Histograms**: Distribution of learned parameters
- **Gradient Flow**: Monitoring gradient magnitudes across layers
- **Learning Rate Schedule**: Adaptive learning rate visualization

### Comparative Analysis
```python
# Compare with PyTorch implementation
import torch.nn as torch_nn

# Custom implementation results
custom_accuracy = nn.evaluate(X_test, y_test)

# PyTorch baseline
torch_model = torch_nn.Sequential(
    torch_nn.Linear(784, 128),
    torch_nn.ReLU(),
    torch_nn.Linear(128, 64),
    torch_nn.ReLU(),
    torch_nn.Linear(64, 10),
    torch_nn.Softmax(dim=1)
)

# Training and comparison...
```

## 🎓 Learning Objectives

After working through this implementation, you will understand:

1. **Neural Network Fundamentals**: How neurons process and combine information
2. **Backpropagation Algorithm**: Mathematical foundation of neural network training
3. **Gradient Descent**: Optimization principles and variants
4. **Activation Functions**: Role in network expressiveness and training dynamics
5. **Regularization**: Techniques for preventing overfitting
6. **Hyperparameter Tuning**: Impact of various configuration choices
7. **Deep Learning Frameworks**: Appreciation for high-level libraries

## 📁 Files Description

- `mlp.ipynb`: Complete neural network implementation and experiments
- Contains detailed explanations of:
  - Mathematical derivations
  - Implementation details
  - Training procedures
  - Performance analysis
  - Comparison with established frameworks

## 🔍 Advanced Topics

### Optimization Enhancements
- **Momentum**: Accelerate convergence and escape local minima
- **Adam Optimizer**: Combine momentum with adaptive learning rates
- **Learning Rate Scheduling**: Dynamic adjustment for better convergence
- **Gradient Clipping**: Prevent exploding gradients in deep networks

### Architectural Improvements
- **Batch Normalization**: Normalize inputs to each layer
- **Residual Connections**: Skip connections for deeper networks
- **Attention Mechanisms**: Focus on relevant input features
- **Ensemble Methods**: Combine multiple networks for better performance

### Regularization Strategies
- **Dropout Variants**: Spatial dropout, scheduled dropout
- **Data Augmentation**: Increase training data diversity
- **Weight Decay**: L2 regularization implementation
- **Noise Injection**: Add robustness to input variations

## 🚀 Getting Started

1. **Open the Jupyter notebook**: `mlp.ipynb`
2. **Follow the step-by-step implementation**
3. **Run each cell** to see the neural network in action
4. **Experiment with hyperparameters** to understand their effects
5. **Compare results** with PyTorch/TensorFlow implementations
6. **Visualize training progress** and network behavior

## 📚 Educational Value

This implementation serves as an excellent learning resource because it:

- **Demystifies neural networks** by showing every computational step
- **Provides mathematical intuition** behind backpropagation
- **Demonstrates best practices** in neural network implementation
- **Offers hands-on experience** with gradient-based optimization
- **Builds foundation** for understanding advanced deep learning concepts

## 🔬 Experimental Features

The notebook includes experiments with:
- **Different architectures** and their performance impact
- **Activation function comparisons** across various tasks
- **Optimization algorithm benchmarks** for convergence speed
- **Regularization effectiveness** on overfitting prevention
- **Hyperparameter sensitivity analysis** for robust training

---

This Multilayer Perceptron implementation provides a comprehensive foundation for understanding neural networks and serves as a stepping stone to more advanced deep learning architectures and techniques.