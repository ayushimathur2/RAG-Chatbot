import os
import pickle
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import google.generativeai as genai

# Step 1: Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=google_api_key)

# Step 2: Define file paths
FILE_PATH = "rag_chunks.pkl"
INDEX_PATH = "rag_index.faiss"

# Step 3: Load the chunks from the pickle file
def load_chunks():
    try:
        with open(FILE_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"File '{FILE_PATH}' not found. Please run 'processor.py' first.")
        exit()

# Step 4: Embed the chunks and save the index
def embed_chunks():
    chunks = load_chunks()
    
    # Check if chunks are loaded correctly
    if not chunks:
        print("No chunks found in the loaded file.")
        return

    # Initialize the embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

    try:
        print("Generating embeddings. This may take a few minutes...")
        embeddings = embedding_model.embed_documents(chunks)
        print("Embeddings generated successfully.")
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        return

    # Check if embeddings were generated before proceeding
    if 'embeddings' not in locals():
        print("Embeddings could not be generated. Please check the error message above.")
        return

    embeddings_array = np.array(embeddings, dtype='float32') # FAISS needs a specific type of number

    # Create a FAISS index and save it
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

if __name__ == '__main__':
    embed_chunks()