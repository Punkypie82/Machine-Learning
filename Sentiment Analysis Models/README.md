# Sentiment Analysis Models

This directory contains two comprehensive implementations of sentiment analysis systems: a BERT-based model using transformer architecture and a custom neural network using sentence embeddings. Both implementations include complete web applications for interactive sentiment prediction.

## 📋 Overview

Sentiment analysis is a natural language processing task that determines the emotional tone or opinion expressed in text. This collection provides:

- **BERT-based sentiment analysis** using state-of-the-art transformer architecture
- **Custom neural network** with sentence embeddings for lightweight deployment
- **Interactive web applications** built with Streamlit for both models
- **Comprehensive preprocessing pipelines** for text data
- **Performance comparison** between different approaches
- **Production-ready deployment** examples

## 🤖 Models Included

### 1. BERT Sentiment Analysis
Advanced transformer-based model fine-tuned for sentiment classification.

### 2. Simple Sentiment Neural Network
Lightweight neural network using pre-trained sentence embeddings.

## 🔧 Project Structure

```
Sentiment Analysis Models/
├── Bert Sentiment/
│   ├── bert_model.ipynb          # BERT training and evaluation
│   ├── bert_api.py               # Streamlit web application
│   └── Model_Files/              # Saved model artifacts
│       ├── bert_model/           # Fine-tuned BERT model
│       └── label_encoder.pkl     # Label encoding mappings
└── Simple SentimentNN/
    ├── initial_models.ipynb      # Model development and training
    ├── simple_api.py             # Streamlit web application
    └── Model_Files/              # Saved model artifacts
        ├── sentiment_model.pt    # TorchScript model
        └── label_encoder.pkl     # Label encoding mappings
```

## 🎯 BERT Sentiment Analysis

### Overview
BERT (Bidirectional Encoder Representations from Transformers) represents the state-of-the-art in natural language understanding. This implementation fine-tunes BERT for sentiment classification tasks.

### Key Features

#### Transformer Architecture
- **Bidirectional Context**: Understands context from both directions
- **Attention Mechanism**: Focuses on relevant parts of the input
- **Pre-trained Representations**: Leverages large-scale language modeling
- **Fine-tuning**: Adapts pre-trained model to specific sentiment task

#### Implementation Details
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,  # Positive, Negative, Neutral
    output_attentions=False,
    output_hidden_states=False
)

# Tokenizer for text preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    """Tokenize and encode text for BERT input"""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    return encoding
```

#### Training Process
1. **Data Preprocessing**: Tokenization and encoding for BERT
2. **Fine-tuning**: Adapt pre-trained BERT to sentiment task
3. **Optimization**: Use AdamW optimizer with learning rate scheduling
4. **Evaluation**: Comprehensive metrics on validation set
5. **Model Saving**: Persist fine-tuned model for deployment

### Web Application Features
```python
# Streamlit application (bert_api.py)
import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer

def make_prediction(input_text):
    """Generate sentiment prediction with probabilities"""
    encoding = preprocess_text(input_text)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    
    predicted_class_index = np.argmax(probabilities)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class_label, probabilities

# Streamlit UI
st.title("BERT Sentiment Analysis Model")
input_text = st.text_input("Enter Sentence:")

if input_text:
    predicted_class, sentiment_probs = make_prediction(input_text)
    st.write(f"**Predicted Sentiment:** {predicted_class}")
    
    # Probability visualization
    plt.figure(figsize=(8, 6))
    plt.bar(label_encoder.classes_, sentiment_probs)
    plt.xlabel("Sentiment Class")
    plt.ylabel("Probability")
    plt.title("Sentiment Probabilities")
    st.pyplot(plt)
```

## 🧠 Simple Sentiment Neural Network

### Overview
A lightweight alternative using sentence embeddings and a custom neural network, designed for efficient deployment and faster inference.

### Architecture Components

#### Sentence Embeddings
- **Pre-trained Embeddings**: Uses HuggingFace sentence transformers
- **Multilingual Support**: BERT-based embeddings for various languages
- **Fixed-size Representations**: Consistent input size regardless of text length
- **Semantic Understanding**: Captures meaning beyond word-level features

#### Neural Network Architecture
```python
class SentimentNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3, dropout=0.3):
        super(SentimentNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
```

#### Training Pipeline
```python
def train_sentiment_model(train_loader, val_loader, num_epochs=50):
    model = SentimentNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_acc = evaluate_model(model, val_loader)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model
```

### Web Application Implementation
```python
# Streamlit application (simple_api.py)
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st

# Load models
@st.cache_resource
def load_models():
    sentiment_model = torch.jit.load('Model_Files/sentiment_model.pt')
    sbert_model = SentenceTransformer("HooshvareLab/bert-fa-base-uncased")
    return sentiment_model, sbert_model

def get_sentence_embeddings(sentences, model, batch_size=256):
    """Generate sentence embeddings efficiently"""
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=True)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def make_prediction(input_text):
    """Predict sentiment using sentence embeddings"""
    preprocessed_text = preprocess_text(input_text)
    sentence_embeddings = get_sentence_embeddings([preprocessed_text], sbert_model)
    input_tensor = torch.tensor(sentence_embeddings, dtype=torch.float32)
    
    with torch.no_grad():
        output = sentiment_model(input_tensor)
    
    probabilities = torch.softmax(output, dim=1).numpy().flatten()
    predicted_class_index = np.argmax(probabilities)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_class_label, probabilities
```

## 📊 Performance Comparison

### Model Characteristics

| Feature | BERT Model | Simple NN Model |
|---------|------------|-----------------|
| **Architecture** | Transformer (110M params) | Feedforward NN (~1M params) |
| **Input Processing** | Tokenization + Attention | Sentence Embeddings |
| **Inference Speed** | Slower (~100ms) | Faster (~10ms) |
| **Memory Usage** | High (~500MB) | Low (~50MB) |
| **Accuracy** | Higher (85-90%) | Good (80-85%) |
| **Deployment** | Resource intensive | Lightweight |
| **Interpretability** | Attention weights | Feature importance |

### Use Case Recommendations

#### BERT Model - Best for:
- **High-accuracy requirements**: When precision is critical
- **Complex text understanding**: Nuanced sentiment detection
- **Research applications**: State-of-the-art performance needed
- **Sufficient resources**: GPU availability and memory not constrained

#### Simple NN Model - Best for:
- **Real-time applications**: Fast response requirements
- **Resource constraints**: Limited memory or CPU-only deployment
- **High throughput**: Processing many requests simultaneously
- **Edge deployment**: Mobile or embedded applications

## 🚀 Deployment Options

### Local Deployment
```bash
# BERT Model
cd "Sentiment Analysis Models/Bert Sentiment"
streamlit run bert_api.py

# Simple NN Model
cd "Sentiment Analysis Models/Simple SentimentNN"
streamlit run simple_api.py
```

### Docker Deployment
```dockerfile
# Dockerfile for BERT model
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "bert_api.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Heroku**: Easy deployment with git integration
- **AWS EC2**: Scalable compute instances
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Managed container hosting

## 💻 Usage Examples

### Interactive Web Interface
Both models provide user-friendly web interfaces:

1. **Text Input**: Enter any sentence or paragraph
2. **Instant Prediction**: Get immediate sentiment classification
3. **Probability Visualization**: See confidence scores for each class
4. **Multiple Languages**: Support for various languages (model-dependent)

### API Integration
```python
# Example API usage
import requests

def predict_sentiment(text, model_url):
    response = requests.post(
        f"{model_url}/predict",
        json={"text": text}
    )
    return response.json()

# Usage
result = predict_sentiment("I love this product!", "http://localhost:8501")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")
```

### Batch Processing
```python
def batch_sentiment_analysis(texts, model):
    """Process multiple texts efficiently"""
    results = []
    
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        batch_predictions = model.predict_batch(batch_texts)
        results.extend(batch_predictions)
    
    return results
```

## 🎓 Learning Objectives

After working through these implementations, you will understand:

1. **Transformer Architecture**: How attention mechanisms work in NLP
2. **Transfer Learning**: Fine-tuning pre-trained models for specific tasks
3. **Sentence Embeddings**: Dense representations of text meaning
4. **Model Deployment**: Building web applications for ML models
5. **Performance Trade-offs**: Accuracy vs. speed vs. resource usage
6. **Text Preprocessing**: Tokenization and encoding strategies
7. **Evaluation Metrics**: Assessing sentiment analysis performance

## 📁 Files Description

### Bert Sentiment/
- `bert_model.ipynb`: Complete BERT fine-tuning implementation
- `bert_api.py`: Streamlit web application for BERT model
- `Model_Files/`: Directory containing saved model artifacts

### Simple SentimentNN/
- `initial_models.ipynb`: Neural network development and training
- `simple_api.py`: Streamlit web application for simple model
- `Model_Files/`: Directory containing saved model artifacts

## 🔍 Advanced Features

### Text Preprocessing
```python
def preprocess_text(text):
    """Advanced text preprocessing pipeline"""
    # Remove digits
    text = re.sub(r'[0-9]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text
```

### Model Interpretability
```python
def explain_prediction(model, text, tokenizer):
    """Generate attention-based explanations"""
    inputs = tokenizer(text, return_tensors='pt', return_attention_mask=True)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights
    attention = outputs.attentions[-1].squeeze()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    
    # Visualize attention
    attention_scores = attention.mean(dim=0).numpy()
    
    return list(zip(tokens, attention_scores))
```

### Performance Monitoring
```python
class ModelMonitor:
    def __init__(self):
        self.predictions = []
        self.response_times = []
        self.confidence_scores = []
    
    def log_prediction(self, text, prediction, confidence, response_time):
        self.predictions.append({
            'timestamp': datetime.now(),
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'response_time': response_time
        })
    
    def get_performance_metrics(self):
        return {
            'avg_response_time': np.mean(self.response_times),
            'avg_confidence': np.mean(self.confidence_scores),
            'prediction_distribution': pd.Series(self.predictions).value_counts()
        }
```

## 🚀 Getting Started

1. **Choose a model** based on your requirements (accuracy vs. speed)
2. **Install dependencies** from requirements.txt
3. **Download pre-trained models** or train from scratch using notebooks
4. **Run the web application** using Streamlit
5. **Test with various inputs** to understand model behavior
6. **Deploy to production** using your preferred platform

## 🔬 Future Enhancements

### Model Improvements
- **Multi-label Classification**: Handle multiple sentiments simultaneously
- **Aspect-based Sentiment**: Analyze sentiment for specific aspects
- **Emotion Detection**: Extend beyond positive/negative/neutral
- **Multilingual Models**: Support for multiple languages

### Application Features
- **Batch Upload**: Process multiple texts from files
- **API Endpoints**: RESTful API for programmatic access
- **Real-time Analytics**: Dashboard for prediction monitoring
- **User Feedback**: Collect annotations for model improvement

### Deployment Enhancements
- **Model Versioning**: Track and manage different model versions
- **A/B Testing**: Compare model performance in production
- **Auto-scaling**: Handle variable load automatically
- **Monitoring**: Comprehensive logging and alerting

---

These sentiment analysis implementations provide comprehensive examples of modern NLP techniques while demonstrating practical deployment strategies for real-world applications.