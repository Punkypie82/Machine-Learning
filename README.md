# Machine Learning Models Collection

A comprehensive collection of machine learning implementations covering various algorithms and techniques, from traditional models to deep learning approaches. This repository contains well-documented Jupyter notebooks and Python applications demonstrating practical implementations of ML algorithms.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Included](#models-included)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository serves as a comprehensive learning resource and reference for machine learning implementations. Each model is implemented from scratch (where applicable) alongside comparisons with scikit-learn and other established libraries. The collection includes:

- **Traditional ML Models**: Decision Trees, KNN, Linear Regression, Linear Classification
- **Neural Networks**: Multilayer Perceptrons, Convolutional Neural Networks
- **Unsupervised Learning**: Clustering algorithms (K-Means, DBSCAN)
- **Natural Language Processing**: Sentiment analysis with BERT and custom neural networks

## 📁 Project Structure

```
├── 1. Decision Tree and KNN Models/
│   ├── DecisionTree/          # Custom decision tree implementation
│   └── KNN/                   # K-Nearest Neighbors implementation
├── 2. Linear Models/
│   ├── LinearClassification/  # Linear classification models
│   └── LinearRegression/      # Linear regression implementation
├── 3. Multilayer Perceptron/  # Neural network from scratch
├── 4. CNN/                    # Convolutional Neural Network
├── 5. Clustering/             # Clustering algorithms and analysis
└── Sentiment Analysis Models/
    ├── Bert Sentiment/        # BERT-based sentiment analysis
    └── Simple SentimentNN/    # Custom neural network for sentiment
```

## 🤖 Models Included

### 1. Decision Tree and K-Nearest Neighbors
- **Decision Tree**: Custom implementation using entropy and information gain
- **KNN**: Implementation with various distance metrics
- **Datasets**: Loan approval data, adult census data

### 2. Linear Models
- **Linear Regression**: From-scratch implementation with gradient descent
- **Linear Classification**: Logistic regression and linear classifiers
- **Datasets**: Housing prices, customer churn prediction

### 3. Multilayer Perceptron
- **Custom Neural Network**: Built from scratch using NumPy/PyTorch
- **Features**: Backpropagation, various activation functions, regularization

### 4. Convolutional Neural Network
- **CNN Implementation**: For image classification tasks
- **Dataset**: 102 Flower Species dataset
- **Features**: Data augmentation, transfer learning concepts

### 5. Clustering Analysis
- **K-Means**: Implementation and analysis
- **DBSCAN**: Density-based clustering
- **Evaluation**: Silhouette analysis, elbow method
- **Dataset**: Customer segmentation data

### 6. Sentiment Analysis Models
- **BERT Model**: Fine-tuned BERT for sentiment classification
- **Simple Neural Network**: Custom embedding-based sentiment analysis
- **Features**: Streamlit web applications for both models

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required Python packages (see Requirements section)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd machine-learning-models-collection
```

2. Create a virtual environment:
```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📦 Requirements

### Core Libraries
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Deep Learning
```
torch>=1.9.0
torchvision>=0.10.0
transformers>=4.11.0
sentence-transformers>=2.0.0
```

### Web Applications
```
streamlit>=1.0.0
```

### Additional Libraries
```
umap-learn>=0.5.0
joblib>=1.0.0
scipy>=1.7.0
tqdm>=4.60.0
Pillow>=8.3.0
```

## 💻 Usage

### Running Jupyter Notebooks

Navigate to any model directory and open the corresponding notebook:

```bash
cd "1. Decision Tree and KNN Models/DecisionTree"
jupyter notebook decision_tree.ipynb
```

### Running Streamlit Applications

For sentiment analysis models:

```bash
# BERT Sentiment Analysis
cd "Sentiment Analysis Models/Bert Sentiment"
streamlit run bert_api.py

# Simple Sentiment Analysis
cd "Sentiment Analysis Models/Simple SentimentNN"
streamlit run simple_api.py
```

### Example Usage

Each notebook includes:
- Data loading and preprocessing
- Model implementation
- Training and evaluation
- Comparison with established libraries
- Visualization of results

## ✨ Features

### Educational Value
- **From-scratch implementations** for better understanding
- **Detailed explanations** of algorithms and concepts
- **Step-by-step code** with comprehensive comments
- **Mathematical foundations** explained where applicable

### Practical Applications
- **Real-world datasets** for practical learning
- **Performance comparisons** with established libraries
- **Visualization tools** for better insights
- **Web applications** for model deployment

### Code Quality
- **Clean, readable code** following best practices
- **Modular design** for easy understanding and modification
- **Comprehensive documentation** in notebooks
- **Error handling** and validation

## 📊 Datasets

The repository uses various datasets including:
- **Loan Approval Dataset**: For decision tree classification
- **Adult Census Dataset**: For KNN classification
- **Customer Churn Dataset**: For linear classification
- **Housing Prices Dataset**: For linear regression
- **102 Flower Species**: For CNN image classification
- **Customer Segmentation Data**: For clustering analysis
- **Sentiment Analysis Data**: For NLP models

## 🎓 Learning Objectives

After working through this repository, you will understand:

1. **Algorithm Implementation**: How to build ML algorithms from scratch
2. **Data Preprocessing**: Techniques for cleaning and preparing data
3. **Model Evaluation**: Various metrics and validation techniques
4. **Hyperparameter Tuning**: Optimization strategies for better performance
5. **Visualization**: How to create meaningful plots and charts
6. **Deployment**: Building web applications for model serving

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Ensure code follows existing style and documentation standards
- Add comprehensive comments and documentation
- Include appropriate tests and validation
- Update README files as necessary

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn documentation and examples
- PyTorch tutorials and community
- Hugging Face Transformers library
- Various dataset providers and Kaggle community
- Open source ML community for inspiration and best practices

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

---

**Happy Learning! 🚀**