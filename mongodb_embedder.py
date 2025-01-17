import numpy as np
from pymongo import MongoClient
import os
from TextEmbedder import gemini_embedder
import re

CONTEXT_CHUNK_MAX_WORDS = 170


def clean_text(text):
    # Use regex to find words with at least one alphanumeric character
    cleaned_words = [word for word in text.split() if re.search(r'\w', word)]
# Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text
#"""
# Function to generate and store embeddings
def store_embeddings(blog_collection, embeddings_collection):
    for blog in blog_collection.find():
        print("updating doc id: ", blog['_id'])
        text = blog['content']  # Assuming 'text' contains the paper's content

        """
        break into chunks of 170 words, put up a for loop for each chunk of blogId
        """
        wordslist = clean_text(text).split()

        for i in range(0, len(wordslist), CONTEXT_CHUNK_MAX_WORDS):
            embedding = gemini_embedder.get_embedding(" ".join(wordslist[i: i + CONTEXT_CHUNK_MAX_WORDS]))
            embeddings_collection.update_one({'blogId': blog['_id'], 'chunk_number': i / CONTEXT_CHUNK_MAX_WORDS}, {'$set': {'embedding': embedding}}, upsert = True)
            print(f"done the chunk {i / CONTEXT_CHUNK_MAX_WORDS} for: {blog['_id']}", )



if __name__ == "__main__":
    client = MongoClient(os.environ['MONGODB_PEM'])
    db = client["pem"]
    blog_collection = db["blogs"]
    embeddings_collection = db["blog_embeddings"]
    store_embeddings(blog_collection, embeddings_collection)
#"""
