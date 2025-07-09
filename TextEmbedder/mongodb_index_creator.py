from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
import time
import os

def create_vector_index(collection: Collection, index_name: str, num_dimensions: int = 768, document_belongs_to_a_type= ""):
    """
    Creates a MongoDB Atlas Vector Search index if it does not already exist.

    Args:
        collection: The PyMongo Collection object on which to create the index.
        index_name: The desired name for the vector search index.
        num_dimensions: The number of dimensions for the embedding vectors.
    """

    # 1. Check if the index already exists
    existing_indexes = list(collection.list_search_indexes())

    for index in existing_indexes:
        if index.get('name') == index_name:
            print(f"Search index '{index_name}' already exists. Skipping creation.")

            # Optional: You can also check if the existing index is "READY"
            if index.get('status') == 'READY':
                print(f"Index '{index_name}' is already ready for querying.")
            else:
                print(f"Index '{index_name}' is currently {index.get('status')}. Polling for readiness...")
                # If it exists but is not ready, proceed to the polling loop
                _wait_for_index_ready(collection, index_name)
            return # Exit the function if the index already exists

    # 2. If the index does not exist, proceed to create it
    print(f"Search index '{index_name}' does not exist. Creating it now...")

    fields_arr= [
        {
            "type": "vector",
            "path": "embedding",
            "similarity": "dotProduct", # Or "cosine", "euclidean"
            "numDimensions": num_dimensions,
            "quantization": "scalar" # Or "none"
        },
        {
            "type": "filter",
            "path": "docId"
        }
    ]

    if document_belongs_to_a_type:
        fields_arr.append({
            "type": "filter",
            "path": "type"
        })
    search_index_model = SearchIndexModel(definition={
            "fields": fields_arr
        },
        name=index_name,
        type="vectorSearch"
    )

    try:
        # Create the search index
        result = collection.create_search_index(model=search_index_model)
        print("New search index named " + result + " is building.")

        # Wait for initial sync to complete
        _wait_for_index_ready(collection, index_name)
        print(result + " is ready for querying.")

    except Exception as e:
        print(f"Error creating or waiting for search index '{index_name}': {e}")
        # Depending on the error, you might want to re-raise or handle it differently.

def _wait_for_index_ready(collection: Collection, index_name: str):
    """
    Helper function to poll the index status until it's ready.
    """
    print("Polling to check if the index is ready. This may take some time (up to a few minutes for large indexes).")

    start_time = time.time()
    timeout = 300 # 5 minutes timeout, adjust as needed

    while True:
        indices= None
        try:
            indices = list(collection.list_search_indexes(name=index_name))
            if len(indices) == 1 and indices[0].get("status") == "READY":
                print(f"Index '{index_name}' is READY for querying.")
                break
            elif len(indices) == 0:
                print(f"Warning: Index '{index_name}' not found during polling, might have failed creation or name mismatch.")
                break # Exit if index disappears
            else:
                current_status = indices[0].get("status", "UNKNOWN")
                print(f"Index '{index_name}' status: {current_status}. Waiting...")
        except Exception as e:
            print(f"Error while polling index status: {e}. Retrying...")

        if time.time() - start_time > timeout:
            status= indices[0].get('status') if indices else 'N/A'
            print(f"Timeout: Index '{index_name}' did not become ready within {timeout} seconds. Current status: {status}")
            break # Exit on timeout

        time.sleep(10) # Poll less frequently, every 10 seconds

# --- Example Usage ---
if __name__ == "__main__":

    # Replace with your database and collection names
    DATABASE_NAME = "pembot"
    COLLECTION_NAME = "doc_chunks"
    VECTOR_INDEX_NAME = "test_search"

    # Connect to MongoDB
    mongo_client= None

    try:
        mongo_client = MongoClient(os.environ["MONGODB_PEM"])
        db = mongo_client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        print("Connected to MongoDB successfully.")

        # Define dimensions (e.g., for nomic-embed-text:v1.5)
        EMBEDDING_DIMENSIONS = 768 # Check your model's output dimension

        # Call the function to create the index, with existence check
        create_vector_index(collection, VECTOR_INDEX_NAME, num_dimensions=EMBEDDING_DIMENSIONS)

        # Test calling it again to see the "already exists" message
        create_vector_index(collection, VECTOR_INDEX_NAME, num_dimensions=EMBEDDING_DIMENSIONS)

    except Exception as e:
        print(f"Failed to connect to MongoDB or process: {e}")
    finally:
        if 'mongo_client' in locals() and mongo_client:
            mongo_client.close()
            print("MongoDB connection closed.")
