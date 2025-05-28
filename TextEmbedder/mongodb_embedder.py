from google import genai
import numpy as np
from pymongo import MongoClient
import os
from TextEmbedder import gemini_embedder
import re



def clean_text(text):
    # Use regex to find words with at least one alphanumeric character
    cleaned_words = [word for word in text.split() if re.search(r'\w', word)]
# Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


#"""
# Function to generate and store embeddings
def mongodb_embed(genai_client, collection, content_field_name: str, embeddings_collection, chunk_wordlimit= 170):
    for doc in collection.find():
        print("updating doc id: ", doc['_id'])
        text = doc[content_field_name]  # Assuming 'text' contains the paper's content

        """
        break into chunks of 170 words, put up a for loop for each chunk of docId
        """
        wordslist = clean_text(text).split()

        for i in range(0, len(wordslist), chunk_wordlimit):
            embedding = gemini_embedder.get_embedding(genai_client, " ".join(wordslist[i: i + chunk_wordlimit]))
            embeddings_collection.update_one({'docId': doc['_id'], 'chunk_number': i / chunk_wordlimit}, {'$set': {'embedding': embedding}}, upsert = True)
            print(f"done the chunk {i / chunk_wordlimit} for: {doc['_id']}", )



if __name__ == "__main__":
    client = MongoClient(os.environ['MONGODB_PEM'])
    db = client["pem"]
    collection = db["blogs"]
    embeddings_collection = db["blog_embeddings"]

    api_key = os.environ['GEMINI_API_KEY']
    genai_client= genai.Client(api_key= api_key)

    mongodb_embed(genai_client, collection, 'content', embeddings_collection)
#"""
