import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the saved model
MODEL_PATH = 'emotion_recognition_model.h5'
model = load_model(MODEL_PATH)
df = pd.read_csv('emotion_sentimen_dataset.csv')
# Preprocess the text

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = text.lower() # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stop words
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
labels = pd.get_dummies(df['Emotion'])
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
texts = df['cleaned_text'].tolist()
# Assume 'texts' is the same training text used to fit the tokenizer
# Replace 'texts' with the actual training texts
tokenizer.fit_on_texts(["texts"])  

# Define emotion labels (replace with your actual labels)
emotion_labels = ["anger", "hate", "neutral", "joy"]  # Update based on your dataset

# Function to preprocess the input sentence
def preprocess_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove special characters
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=50)
    return padded_sequence

def predict_emotion(sentence, tokenizer, model):
    # Preprocess the input sentence
    # Use import re and re.sub for regex replacement
    import re
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence) 
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=50)
    
    # Predict
    prediction = model.predict(padded_sequence)
    emotion_idx = np.argmax(prediction)
    emotion_labels = labels.columns  # The one-hot encoded label columns
    return emotion_labels[emotion_idx]

# Streamlit app
st.title("Emotion Recognition from Text")
st.write("Enter a sentence to predict the associated emotion.")

# Input text
input_text = st.text_input("Enter your sentence:", "")

if st.button("Predict Emotion"):
    if input_text.strip():
        emotion = predict_emotion(input_text, tokenizer, model)
        st.success(f"Predicted Emotion: {emotion}")
    else:
        st.error("Please enter a valid sentence.")
