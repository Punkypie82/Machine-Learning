# Convolutional Neural Network (CNN)

This directory contains a comprehensive implementation of Convolutional Neural Networks for image classification tasks. The implementation demonstrates the fundamental concepts of computer vision and deep learning applied to visual recognition problems.

## 📋 Overview

Convolutional Neural Networks are specialized deep learning architectures designed for processing grid-like data such as images. This implementation provides:

- **Complete CNN architecture** from scratch using PyTorch
- **Image classification pipeline** with data preprocessing and augmentation
- **Advanced CNN techniques** including transfer learning concepts
- **Comprehensive evaluation** with visualization and analysis tools
- **Performance optimization** strategies for efficient training

## 🖼️ Dataset: 102 Flower Species Classification

### Dataset Overview
The implementation uses the Oxford 102 Flower Species dataset, a challenging computer vision benchmark:

#### Dataset Characteristics
- **Classes**: 102 different flower species
- **Images**: High-resolution color images with varying backgrounds
- **Complexity**: Natural variations in lighting, pose, and occlusion
- **Size**: Thousands of images across all categories
- **Challenge**: Fine-grained classification with subtle inter-class differences

#### Data Source
- **Origin**: Kaggle dataset (https://www.kaggle.com/datasets/mehmetors/102-flower-dataset)
- **Format**: JPEG images with corresponding labels
- **Quality**: High-resolution images suitable for deep learning
- **Diversity**: Wide variety of flower types, colors, and compositions

## 🏗️ CNN Architecture

### Network Design
The implementation includes multiple CNN architectures:

#### Basic CNN Architecture
```python
class BasicCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
```

#### Advanced CNN Features
- **Multiple Convolutional Layers**: Feature extraction at different scales
- **Pooling Operations**: Spatial dimension reduction and translation invariance
- **Batch Normalization**: Stable training and faster convergence
- **Dropout Regularization**: Overfitting prevention
- **Skip Connections**: Gradient flow improvement (ResNet-style)

### Key Components

#### Convolutional Layers
- **Feature Maps**: Learn spatial hierarchies of features
- **Kernels/Filters**: Detect edges, textures, and complex patterns
- **Stride and Padding**: Control output dimensions and border handling
- **Activation Functions**: ReLU for non-linearity introduction

#### Pooling Layers
- **Max Pooling**: Retain strongest activations
- **Average Pooling**: Smooth feature representations
- **Global Average Pooling**: Reduce overfitting in final layers
- **Adaptive Pooling**: Fixed output size regardless of input dimensions

#### Fully Connected Layers
- **Feature Integration**: Combine spatial features for classification
- **Classification Head**: Final decision-making layers
- **Regularization**: Dropout and weight decay for generalization

## 🔧 Implementation Features

### Data Pipeline
```python
# Data transformations and augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

### Training Pipeline
```python
def train_model(model, train_loader, val_loader, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        
        # Validation and metrics tracking...
```

## 📊 Advanced Techniques

### Data Augmentation
- **Geometric Transformations**: Rotation, flipping, scaling
- **Color Augmentation**: Brightness, contrast, saturation adjustments
- **Noise Injection**: Gaussian noise for robustness
- **Cutout/Mixup**: Advanced augmentation techniques
- **Test Time Augmentation**: Multiple predictions for better accuracy

### Regularization Methods
- **Dropout**: Random neuron deactivation during training
- **Batch Normalization**: Normalize layer inputs for stable training
- **Weight Decay**: L2 regularization on model parameters
- **Early Stopping**: Prevent overfitting through validation monitoring
- **Data Augmentation**: Increase effective dataset size

### Optimization Strategies
- **Learning Rate Scheduling**: Adaptive learning rate adjustment
- **Gradient Clipping**: Prevent exploding gradients
- **Warm-up Strategies**: Gradual learning rate increase
- **Optimizer Selection**: Adam, SGD, RMSprop comparison
- **Batch Size Tuning**: Memory vs. convergence trade-offs

## 📈 Performance Analysis

### Training Metrics
```python
# Comprehensive evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

### Visualization Tools
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrix**: Detailed classification performance
- **Feature Maps**: Visualize learned convolutional filters
- **Grad-CAM**: Highlight important image regions
- **Sample Predictions**: Visual inspection of model performance

### Performance Results
Typical performance metrics on the 102 Flower dataset:
- **Training Accuracy**: 85-95% (depending on architecture)
- **Validation Accuracy**: 75-85% (with proper regularization)
- **Test Accuracy**: 70-80% (realistic generalization performance)
- **Top-5 Accuracy**: 90-95% (more lenient metric for fine-grained classification)

## 💻 Usage Examples

### Basic Training
```python
# Initialize model and data loaders
model = FlowerCNN(num_classes=102)
model = model.to(device)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
history = train_model(model, train_loader, val_loader, num_epochs=50)

# Evaluate performance
test_metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
```

### Advanced Configuration
```python
# Advanced model with transfer learning concepts
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=102):
        super(AdvancedCNN, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Additional blocks...
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
```

## 🔍 Computer Vision Concepts

### Feature Learning Hierarchy
1. **Low-level Features**: Edges, corners, textures (early layers)
2. **Mid-level Features**: Shapes, patterns, object parts (middle layers)
3. **High-level Features**: Complete objects, semantic concepts (deep layers)
4. **Classification**: Final decision based on learned representations

### Spatial Hierarchies
- **Local Receptive Fields**: Small regions in early layers
- **Growing Receptive Fields**: Larger regions in deeper layers
- **Translation Invariance**: Robustness to object position
- **Scale Invariance**: Handling objects of different sizes

### Transfer Learning Concepts
- **Feature Reusability**: Lower layers learn general features
- **Domain Adaptation**: Fine-tuning for specific tasks
- **Progressive Feature Complexity**: From edges to semantic concepts
- **Computational Efficiency**: Leverage pre-trained representations

## 🎓 Learning Objectives

After working through this CNN implementation, you will understand:

1. **Convolutional Operations**: How filters detect features in images
2. **Pooling Mechanisms**: Spatial dimension reduction and invariance
3. **Feature Hierarchies**: Progressive abstraction in deep networks
4. **Image Classification Pipeline**: End-to-end computer vision workflow
5. **Data Augmentation**: Techniques for improving generalization
6. **Regularization in CNNs**: Preventing overfitting in deep networks
7. **Performance Evaluation**: Metrics and visualization for image classification

## 📁 Files Description

- `cnn.ipynb`: Complete CNN implementation and flower classification
- Includes:
  - Dataset loading and preprocessing
  - CNN architecture definition
  - Training loop implementation
  - Performance evaluation and visualization
  - Comparison with established architectures

## 🚀 Getting Started

1. **Download the dataset** from the provided Kaggle link
2. **Extract the dataset** to the appropriate directory
3. **Open the Jupyter notebook**: `cnn.ipynb`
4. **Follow the step-by-step implementation**
5. **Run training** with different hyperparameters
6. **Analyze results** and visualizations
7. **Experiment** with different architectures

## 🔬 Experimental Extensions

### Architecture Variations
- **Different Depths**: Experiment with network depth effects
- **Filter Sizes**: Compare 3x3 vs 5x5 vs 7x7 kernels
- **Skip Connections**: Implement ResNet-style connections
- **Attention Mechanisms**: Add spatial attention modules

### Advanced Techniques
- **Transfer Learning**: Use pre-trained ImageNet models
- **Ensemble Methods**: Combine multiple CNN predictions
- **Progressive Resizing**: Train with increasing image sizes
- **Mixed Precision Training**: Faster training with reduced memory

### Evaluation Enhancements
- **Cross-Validation**: Robust performance estimation
- **Error Analysis**: Detailed failure case investigation
- **Interpretability**: Visualize what the network learns
- **Robustness Testing**: Evaluate on corrupted images

---

This CNN implementation provides a comprehensive foundation for understanding computer vision and deep learning, serving as a practical introduction to modern image classification techniques while maintaining educational clarity and depth.