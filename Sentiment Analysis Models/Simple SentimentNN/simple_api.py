import re
import torch
import joblib
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Select the device (MPS for Apple M1/M2 chips or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the label encoder to map predictions back to sentiment classes
label_encoder = joblib.load('Model_Files/label_encoder.pkl')

# Load the pre-trained TorchScript model and SentenceTransformer model once into session state
if 'model' not in st.session_state:
    st.session_state.model = torch.jit.load('Model_Files/sentiment_model.pt').to(device)

if 'sbert_model' not in st.session_state:
    model_name = "HooshvareLab/bert-fa-base-uncased"
    st.session_state.sbert_model = SentenceTransformer(model_name)
    st.session_state.sbert_model.to(device)

# Preprocessing function to remove digits from the text
def preprocess_text(text):
    return re.sub(r'[0-9]', '', text)

# Function to get sentence embeddings using SentenceTransformer model
def get_sentence_embeddings(sentences, model, batch_size=256):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        embeddings = model.encode(batch, show_progress_bar=True, device=device)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Function to make sentiment predictions from the input text
def make_prediction(input_text):
    preprocessed_text = preprocess_text(input_text)  # Clean input text
    sentence_embeddings = get_sentence_embeddings([preprocessed_text], st.session_state.sbert_model)
    input_tensor = torch.tensor(sentence_embeddings, dtype=torch.float32).to(device)
    
    # Perform inference and get model output
    with torch.no_grad():
        output = st.session_state.model(input_tensor)

    probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()  # Get probabilities
    predicted_class_index = np.argmax(probabilities)  # Find the class with highest probability
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]  # Convert to label

    return predicted_class_label, probabilities  # Return predicted label and probabilities

# Streamlit UI
st.title("Simple Sentiment Analysis Model")

# Input: Sentence from the user
input_text = st.text_input("Enter Sentence:")

# If input exists, make the prediction and display results
if input_text:
    predicted_sentiment, sentiment_probs = make_prediction(input_text)
    
    # Display the predicted sentiment class above the chart
    st.write(f"**Predicted Sentiment:** {predicted_sentiment}")

    # Plot the bar chart of probabilities for each sentiment class
    plt.figure(figsize=(8, 6))
    plt.bar(label_encoder.classes_, sentiment_probs)
    plt.xlabel("Sentiment Class")
    plt.ylabel("Probability")
    plt.title("Sentiment Probabilities")
    st.pyplot(plt)
