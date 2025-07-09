from pathlib import Path
import uuid
from google import genai
from google.genai import types
from pymongo import MongoClient
import os
import re
import time
import numpy as np

from pembot.AnyToText.convertor import chunk_text
from pembot.utils.string_tools import make_it_an_id



def clean_text(text):
    # Use regex to find words with at least one alphanumeric character
    cleaned_words = [word for word in text.split() if re.search(r'\w', word)]
# Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text



def search_within_document(
    db_client,
    aggregate_query_embedding,
    document_name_id: str,
    limit: int = 5,
    index_name: str = "test_search",
    embeddings_collection_name: str= "doc_chunks",
    document_belongs_to_a_type = "",
):
    """
    Performs a vector similarity search within the chunks of a specific document
    in the 'embeddings_collection' MongoDB collection.

    Args:
        db_client: An initialized PyMongo Database instance.
        aggregate_query_embedding: The np.mean of queries vectors of your search query.
        document_name_id: This will be used to filter by the 'docId'.
        limit: The maximum number of similar chunks to return.
        index_name: The name of your MongoDB Atlas Vector Search index.
                    You MUST have a vector search index created on the 'embedding' field
                    of the 'embeddings_collection' collection for this to work efficiently.
        document_belongs_to_a_type: When search spaces intersect for different docIds, such that docId is an array field,

    Returns:
        A list of dictionaries, where each dictionary represents a matching chunk
        from the specified document, including its text, docId, and score.
    """
    embeddings_collection = db_client[embeddings_collection_name]

    print(f"Searching within document (docId: {document_name_id})...")
    # print(f" filter (slug: {document_belongs_to_a_type})...")

    # MongoDB Atlas Vector Search aggregation pipeline
    # The 'path' should point to the field containing the embeddings.
    # The 'filter' stage is crucial for searching within a specific document.
    #
    project_dict= {
        '_id': 0,
        'docId': 1,
        'chunk_number': 1,
        'chunk_text': 1,
        'score': { '$meta': 'vectorSearchScore' } # Get the similarity score
    }

    if document_belongs_to_a_type:
        project_dict['type']= 1

    pipeline = [
        {
            '$vectorSearch': {
                'queryVector': aggregate_query_embedding,
                'path': 'embedding',

                #number of nearest neighbors to consider
                'numCandidates': 100,
                'limit': limit,
                'index': index_name,

                #filter to search only within the specified document
                'filter':
                    { "type": {"$in": [document_belongs_to_a_type ]} } if document_belongs_to_a_type else
                    { 'docId': document_name_id }
            }
        },

        # to exclude the MongoDB internal _id
        {
            '$project': project_dict
        }
    ]

    # print("sesraching now:")
    results = list(embeddings_collection.aggregate(pipeline))
    # print("search results: ", results)

    if not results:
        print(f"No relevant chunks found for document '{document_name_id}' with the given query.")
    else:
        print(f"Found {len(results)} relevant chunks in document '{document_name_id}':")
        for i, res in enumerate(results):
            print(f"  Result {i+1} (Score: {res['score']:.4f}):")
            print(f"    Chunk Number: {res['chunk_number']}")
            print(f"    Text: '{res['chunk_text'][:100]}...'") # Print first 100 chars
            print("-" * 30)

    return results



def process_document_and_embed(
    db_client,
    llm_client,
    inference_client,
    file_path: Path,
    chunk_size: int,
    embedding_model: str = 'BAAI/bge-en-icl',
    embeddings_collection_name= "doc_chunks",
    use_custom_id: str | None = None,
    use_custom_input: str | None = None,
    document_belongs_to_a_type= "",
    type_info= []
) -> list[dict]:
    """
    Processes an input document by chunking its text, generating embeddings using
    Ollama's specified embedding model, and storing these embeddings and chunks
    in a MongoDB collection.

    Args:
        db_client: An initialized PyMongo Database instance.
        file_path: The original path of the document being processed.
                   This path will be used to create a sanitized ID for the
                   document.
        chunk_size: The desired chunk size in words for text segmentation.
        embedding_model: The name of the Ollama embedding model to use.
    """

    input_text= None
    if use_custom_input is not None:
        input_text= use_custom_input
    else:
        # Read the input text from the file
        with open(str(file_path), "r") as md_file:
            input_text = md_file.read()


    document_name_id= None
    if use_custom_id is not None:
        document_name_id= use_custom_id
    else:
        # Create a valid ID for the document from the file name (without extension)
        file_root = os.path.splitext(file_path.name)[0]
        document_name_id = make_it_an_id(file_root)

    # Reference the MongoDB collection where chunks will be stored
    # This single collection will serve as the global 'embeddings_collection'
    # and document-specific data can be queried using 'docId'.
    embeddings_collection = db_client[embeddings_collection_name]

    # Check if this document's embeddings already exist in MongoDB
    # We check if any document with this docId exists in the 'embeddings_collection' collection.
    an_existing_chunk= embeddings_collection.find_one({'docId': document_name_id})
    if an_existing_chunk:
        print(f"Document '{file_path.name}' (ID: {document_name_id}) already processed. Skipping.")
        return [an_existing_chunk]

    print(f"Processing document '{file_path.name}' (ID: {document_name_id})...")

    embed_locally= False
    try:
        models = llm_client.list()

        for model in models.models:
            if model.model == embedding_model:
                embed_locally= True
    except Exception as e:
        print("local model list error: ", e)


    # Chunk the input text into smaller segments
    chunks = chunk_text(input_text, chunk_size)
    print(f"Text chunked into {len(chunks)} segments.")

    # Process each chunk: generate embedding and add to MongoDB
    res = []
    for i, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)} for document '{file_path.name}'...")
            # Generate embedding using the specified Ollama model
            print("embedding_model is: ", embedding_model)
            print("if statement is", 'gemini' in embedding_model)

            if 'gemini' in embedding_model:

                client = genai.Client(api_key= os.environ['GEMINI_API_KEY'])
                result = client.models.embed_content(
                        model= embedding_model,
                        contents= chunk,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                if result is not None and result.embeddings:
                    embedding= result.embeddings[0].values
                else:
                    raise ValueError("Gemini not givingz embeddingzzz")

                # API safety
                time.sleep(1)

            elif embed_locally:
                response = llm_client.embeddings(model=embedding_model, prompt=chunk)
                embedding= response['embedding']

            else:
                embedding = inference_client.feature_extraction(chunk, model=embedding_model)

                # API rate limiting safety
                time.sleep(1)
                print("zzzzzzzzz")

            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                raise TypeError("Embedding is not a list or numpy array. Cannot store in MongoDB.")


            # Generate a random suffix for unique chunk IDs
            random_suffix = uuid.uuid4().hex[:8]

            # Create the unique ID for the chunk for the global 'embeddings_collection' collection
            # This ID can be stored in the document if needed for external reference
            chunk_id_global = f"{document_name_id}_chunk_{i + 1}_{random_suffix}"
            # Create the unique ID for the chunk (local to the document's chunking)
            chunk_id_doc_specific = f"chunk_{i}_{random_suffix}"

            # Store the chunk data in MongoDB using update_one with upsert=True
            # This will insert a new document if 'docId' and 'chunk_number' don't match,
            # or update an existing one if they do.
            doc_set = {
                'chunk_text': chunk,
                'embedding': embedding,
                'chunk_id_global': chunk_id_global,
                'chunk_id_doc_specific': chunk_id_doc_specific,
            }


            # TBD: this is NOT pushing array, this is creating a "$push" field with type: "" object

            if len(type_info) > 0:
                embeddings_collection.update_one(
                    {'docId': document_name_id, 'chunk_number': i + 1},
                    {
                        '$set': doc_set,
                        '$push': {
                            "type": type_info
                        }
                    },
                    upsert=True
                )
            else:

                embeddings_collection.update_one(
                    {'docId': document_name_id, 'chunk_number': i + 1},
                    {'$set': doc_set},
                    upsert=True
                )
            print(f"Successfully stored chunk {i+1} for '{file_path.name}' in MongoDB.")
            res.append({**doc_set, "docId": document_name_id, "chunk_number": i + 1})

        except Exception as e:
            print(f"Error processing chunk {i+1} for '{file_path.name}': {e}")
            # Continue to the next chunk even if one fails
            continue

    print(f"Finished processing document '{file_path.name}'. All chunks embedded and stored in MongoDB.")
    return res




if __name__ == "__main__":
    client = MongoClient(os.environ['MONGODB_PEM'])
    db = client["pem"]
    collection = db["blogs"]
    embeddings_collection = db["blog_embeddings"]

    api_key = os.environ['GEMINI_API_KEY']
