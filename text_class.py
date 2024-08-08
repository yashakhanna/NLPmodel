import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Load the trained model
model = load_model('C:/Users/dhill/VS-Code/text_classification_model.h5')

# Load tokenizer (you should save and load your tokenizer as well)
# For simplicity, we are assuming it is already available
tokenizer = Tokenizer()  # Update this with your actual tokenizer

# Define text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\\n', '', text)
    text = re.sub(f'[{string.punctuation}]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Streamlit app
st.title('Text Classification App')
st.write('Enter the text you want to classify:')

user_input = st.text_area('Text Input')

if st.button('Classify'):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([processed_text])
        max_length = model.input_shape[1]
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
        
        # Make predictions
        predictions = model.predict(padded_sequence)
        discrimination_prob = predictions[0][0][0]
        group_probs = predictions[1][0]

        # Display the results
        st.write(f'Discrimination Probability: {discrimination_prob:.4f}')
        st.write('Group Probabilities:')
        for i, prob in enumerate(group_probs):
            st.write(f'Group {i}: {prob:.4f}')
    else:
        st.write('Please enter some text.')

