import os
from google import genai
from pymongo import MongoClient
from TextEmbedder import gemini_embedder, mongodb_embedder
from os import environ
import re


def vectory_query_run(genai_client, docs_collection, embeddings_collection, search_string, database, index_name, chunksize= 170):

    query_embedding = gemini_embedder.get_embedding(genai_client, search_string)

    # Define the aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "index": f"{index_name}",
                "path": "embedding",
                "numCandidates": 3,
                "limit": 3,
                "k": 3  # Number of results to return
            }
        }
    ]

    # Execute the aggregation pipeline
    results = list(embeddings_collection.aggregate(pipeline))
    for i, res in enumerate(results):
        doc = docs_collection.find_one({"_id": res['docId']})

        wordslist = mongodb_embedder.clean_text(doc['content']).split()

        cn = int(res['chunk_number'])
        search_hit = " ".join(wordslist[cn * chunksize: (cn+1) * chunksize])

        print(f"RESULT #{i + 1}")
        print(search_hit)
        print("\n\n")
        


if __name__ == "__main__":
    mc = MongoClient(environ['MONGODB_PEM'])

    api_key = os.environ['GEMINI_API_KEY']
    genai_client= genai.Client(api_key= api_key)

    try:
        mc.admin.command('ping')  # This is a simple way to check the connection
        database = mc["pem"]

        while True:
            query = input("enter search query [Enter 'exit' to exit]: ")
            if query.lower() == 'exit':
                print('bye!')
                break
            docs_collection= database["blogs"]
            embeddings_collection= database["blog_embeddings"]
            vectory_query_run(genai_client, docs_collection, embeddings_collection, query, database, index_name= 'vector_index_blog_chunks')

    finally:
        mc.close()

