import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-model4")
model = GPT2LMHeadModel.from_pretrained("./fine-tuned-model4")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Load the FAISS index
index = faiss.read_index("chatbot_faiss.index")

# Load the input-response pairs
parsed_data = pd.read_csv("chatbot_conversations.csv")

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the most similar input text and its response
def find_response(user_input):
    user_embedding = embedder.encode([user_input])
    _, indices = index.search(user_embedding, k=1)
    response = parsed_data['response_texts'][indices[0][0]]
    return response

# Function to generate a response using RAG
def generate_response(user_input):
    retrieved_response = find_response(user_input)
    input_text = f"User: {user_input}\nChatbot: {retrieved_response}"
    generated_response = generator(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']
    return generated_response

# Streamlit app
def app():
    st.title("Support Chatbot")

    user_input = st.text_input("You: ")
    if user_input:
        response = generate_response(user_input)
        st.text_area("Chatbot:", response)

if __name__ == '__main__':
    app()
