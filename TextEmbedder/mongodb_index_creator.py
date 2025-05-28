
def create_vector_index(mongo_client, collection, index_name, num_dimensions: int):
    try:

        index = {
            "name": f"{index_name}",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "similarity": "dotProduct",
                        "numDimensions": num_dimensions
                    }
                ]
            }
        }

        # Create the search index
        result = collection.create_index(index)
        print(result)
    finally:
        mongo_client.close()

if __name__ == "__main__":
    print("mongodb index creation file")

