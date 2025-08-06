import streamlit as st
import os
from dotenv import load_dotenv
import pickle
import faiss
import numpy as np
from google.generativeai import GenerativeModel
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings # This is needed for a specific method later

# Step 1: Load the secret key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY in your .env file.")
    st.stop()

# Configure the Generative AI client with the API key
genai.configure(api_key=google_api_key)

# Step 2: Load the prepared data (the library)
try:
    with open('rag_chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    index = faiss.read_index('rag_index.faiss')
except FileNotFoundError as e:
    st.error(f"Error loading RAG components: {e}. Please run scraper.py, processor.py, and embedder.py in order first.")
    st.stop()

# Step 3: Set up the AI brain
generative_model = GenerativeModel("models/gemini-1.5-flash")

# Step 4: Build the Streamlit webpage
st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 Website Q&A Chatbot")
st.write("Ask me anything about the scraped website!")

# Step 5: Create a "memory" for the chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# Step 6: Show the past conversation
for role, content in st.session_state.messages:
    if role == "user":
        st.markdown(f"**🧑 You:** {content}")
    else:
        st.markdown(f"**🤖 Bot:** {content}")

# Step 7: Get new input from the user
user_query = st.chat_input("Your question:")

if user_query:
    # A. Show the user's question and save it in memory
    st.markdown(f"**🧑 You:** {user_query}")
    st.session_state.messages.append(("user", user_query))

    # B. Find the most relevant information using the FAISS index
    with st.spinner("Thinking..."):
        # Use the direct API call for embedding to avoid the RuntimeError
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=user_query
            )
            query_embedding = response['embedding']
        except Exception as e:
            st.error(f"An error occurred while embedding the query: {e}")
            st.stop()
        
        query_embedding_array = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding_array, k=3)
        retrieved_chunks = " ".join([chunks[i] for i in indices[0]])

        # C. Create the final "instruction" for the AI
        prompt = f"Answer the question based only on the following text:\n\n{retrieved_chunks}\n\nQuestion: {user_query}\nAnswer:"

        # D. Get the answer from the Gemini AI
        try:
            response = generative_model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"An error occurred while generating the response: {e}"

        # E. Show the AI's answer and save it in memory
        st.markdown(f"**🤖 Bot:** {answer}")
        st.session_state.messages.append(("bot", answer))