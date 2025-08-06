import re
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_and_chunk(filepath='scraped_data.txt', chunk_size=500, chunk_overlap=50):
    """
    This function cleans text from a file and chops it into smaller, smarter chunks.
    """
    print("Chef is getting ready to cook...")

    try:
        # Step 1: Get the messy ingredient (text) from the file.
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Step 2: Clean the messy ingredient.
        # This line removes extra spaces and cleans up weird characters.
        text = re.sub(r'\s+', ' ', text).strip()

        # Step 3: Chop the text into pieces using the smart tool.
        # This tool tries to chop at good spots like double newlines or spaces.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  
            length_function=len,
        )
        chunks = text_splitter.split_text(text)


        # This saves the list of chunks to a file called 'rag_chunks.pkl'.
        with open('rag_chunks.pkl', 'wb') as f:
            pickle.dump(chunks, f)

        print(f"✅ Chef is done! Cleaned and chopped text into {len(chunks)} pieces.")
        
    except FileNotFoundError:
        print(f"❌ Oops! The file '{filepath}' is missing. Did you run scraper.py first?")
        
    except Exception as e:
        print(f"❌ A cooking error happened: {e}")

if __name__ == "__main__":
    
    clean_and_chunk()