import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load the API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("Error: GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=google_api_key)
    print("Listing all available models that support 'generateContent':")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")