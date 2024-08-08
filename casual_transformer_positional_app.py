import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
from tensorflow.keras.utils import custom_object_scope

# Define custom layers and register them
@tf.keras.utils.register_keras_serializable()
class RelativePositionalEncoding(Layer):
    def __init__(self, max_seq_len, depth, **kwargs):
        super(RelativePositionalEncoding, self).__init__(**kwargs)
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.embedding = Embedding(input_dim=2 * max_seq_len - 1, output_dim=depth)

    def call(self, x):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        range_vec = tf.range(seq_len)
        range_mat = tf.reshape(range_vec, (-1, 1)) - tf.reshape(range_vec, (1, -1))
        range_mat = range_mat + (self.max_seq_len - 1)
        return self.embedding(range_mat)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'depth': self.depth,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class CausalSelfAttention(Layer):
    def __init__(self, num_heads, key_dim, max_seq_len, **kwargs):
        super(CausalSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.max_seq_len = max_seq_len
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.relative_position_encoding = RelativePositionalEncoding(max_seq_len, key_dim)

    def call(self, inputs, **kwargs):
        seq_len = tf.shape(inputs)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, 0)
        pos_encoding = self.relative_position_encoding(inputs)
        return self.attention(inputs, inputs, attention_mask=mask)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'max_seq_len': self.max_seq_len,
        })
        return config

# Important NLTK data packages downloading
nltk.download('punkt')
nltk.download('wordnet')

# Define constants
vocab_size = 5500
max_seq_len = 100

# Load pre-trained tokenizer
tokenizer = Tokenizer(num_words=vocab_size)

# Load pre-trained label encoder
lbl_encoder = LabelEncoder()

# Define a function to preprocess the text input
def text_preprocess(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text) 
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    return ''

# Function to load the model with custom objects
def load_custom_model(model_path):
    custom_objects = {
        'RelativePositionalEncoding': RelativePositionalEncoding,
        'CausalSelfAttention': CausalSelfAttention
    }
    with custom_object_scope(custom_objects):
        return load_model(model_path)

# Streamlit app
st.title("Sentiment Analysis with Causal Transformer")
st.markdown("Enter a tweet and get its sentiment prediction:")

# Input text box
user_input = st.text_area("Enter your comment here:", "")

if st.button("Predict"):
    if user_input:
        # Preprocess the input text
        clean_text = text_preprocess(user_input)
        
        # Tokenize and pad the input text
        sequence = tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_seq_len)
        
        # Load your trained model with custom objects
        model = load_custom_model(r"C:\Users\dhill\VS-Code\prabh\casual_transformer_positional_model_1.h5")

        # Predict the sentiment
        prediction_prob = model.predict(padded_sequence)
        prediction = np.argmax(prediction_prob, axis=1)
        sentiment = lbl_encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f"Predicted Sentiment: **{sentiment[0]}**")
        
        # Visualize the probabilities
        st.write("Prediction Probabilities:")
        p
