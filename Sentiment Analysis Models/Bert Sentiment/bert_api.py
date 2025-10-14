import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the saved model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained('Model_Files/bert_model')
tokenizer = BertTokenizer.from_pretrained('Model_Files/bert_model')
label_encoder = joblib.load('Model_Files/label_encoder.pkl')

# Set device (MPS for Apple M1/M2 chips or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Preprocessing function to tokenize input
def preprocess_text(text):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    return encoding

# Function to make predictions
def make_prediction(input_text):
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

# Input field
input_text = st.text_input("Enter Sentence:")

if input_text:
    predicted_class, sentiment_probs = make_prediction(input_text)
    
    # Display the predicted sentiment class
    st.write(f"**Predicted Sentiment:** {predicted_class}")
    
    # Display the probability bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(label_encoder.classes_, sentiment_probs)
    plt.xlabel("Sentiment Class")
    plt.ylabel("Probability")
    plt.title("Sentiment Probabilities")
    st.pyplot(plt)
