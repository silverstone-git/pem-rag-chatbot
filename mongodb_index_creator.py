from pymongo import MongoClient
from os import environ


def run():
    client = MongoClient(environ['MONGODB_PEM'])
    try:
        database = client["pem"]
        collection = database["blogs"]

        index = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "similarity": "dotProduct",
                        "numDimensions": 768
                    }
                ]
            }
        }

        # Create the search index
        result = collection.create_index(index)
        print(result)
    finally:
        client.close()

if __name__ == "__main__":
    run()

