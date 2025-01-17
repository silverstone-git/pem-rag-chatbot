from pymongo import MongoClient
from TextEmbedder import gemini_embedder
from os import environ
import re

CHUNK_SIZE = 170



def clean_text(text):
    # Use regex to find words with at least one alphanumeric character
    cleaned_words = [word for word in text.split() if re.search(r'\w', word)]
# Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


def run(search_string, database):
    collection = database["blog_embeddings"]

    query_embedding = gemini_embedder.get_embedding(search_string)  # Get the embedding for the search string

    # Define the aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "index": 'vector_index_blog_chunks',
                "path": "embedding",
                "numCandidates": 3,
                "limit": 3,
                "k": 3  # Number of results to return
            }
        }
    ]

    # Execute the aggregation pipeline
    results = list(collection.aggregate(pipeline))
    blog_collection = database["blogs"]
    for i, res in enumerate(results):
        blog = blog_collection.find_one({"_id": res['blogId']})

        wordslist = clean_text(blog['content']).split()

        cn = int(res['chunk_number'])
        search_hit = " ".join(wordslist[cn * CHUNK_SIZE: (cn+1) * CHUNK_SIZE])

        print(f"RESULT #{i + 1}")
        print(search_hit)
        print("\n\n")
        


if __name__ == "__main__":
    client = MongoClient(environ['MONGODB_PEM'])

    try:
        client.admin.command('ping')  # This is a simple way to check the connection
        database = client["pem"]

        while True:
            query = input("enter search query [Enter 'exit' to exit]: ")
            if query.lower() == 'exit':
                print('bye!')
                break
            run(query, database)

    finally:
        client.close()

