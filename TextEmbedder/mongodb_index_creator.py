from pymongo.operations import SearchIndexModel
import time

def create_vector_index(mongo_client, collection, index_name, num_dimensions: int= 3072):
    search_index_model= SearchIndexModel(definition= {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "similarity": "dotProduct",
                    "numDimensions": num_dimensions,
                    "quantization": "scalar"
                }
            ]
        },

        name= index_name,
        type= "vectorSearch"
    )


    # Create the search index
    result = collection.create_search_index(model=search_index_model)
    print("New search index named " + result + " is building.")

    # Wait for initial sync to complete
    print("Polling to check if the index is ready. This may take up to a minute.")
    predicate=None
    if predicate is None:
      predicate = lambda index: index.get("queryable") is True
    while True:
      indices = list(collection.list_search_indexes(result))
      if len(indices) and predicate(indices[0]):
        break
      time.sleep(5)
    print(result + " is ready for querying.")


if __name__ == "__main__":
    print("mongodb index creation file")

