import uuid
import random
from pembot.AnyToText.convertor import Convertor
from TextEmbedder.chromadb_upload import add

def chunk_text(text, chunk_size=500, overlap_size=50):
    """
    Chunks a given text into smaller pieces with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk (in characters).
        overlap_size (int): The number of characters to overlap between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap_size)
        if start < 0:  # Handle cases where overlap_size is greater than chunk_size
            start = 0
    return chunks

if __name__ == "__main__":
    file_path = "/home/cyto/dev/pem-rag-chatbot/hcltech_ai_engg.md"
    document_common_id = "hcltech_ai_engg_doc"  # Common part for all IDs from this document

    try:
        with open(file_path, "r") as md_file:
            full_document_text = md_file.read()

        # Chunk the document
        chunks = chunk_text(full_document_text, chunk_size=1000, overlap_size=100) # Adjust chunk_size and overlap_size as needed

        # Add chunks to ChromaDB
        for i, chunk in enumerate(chunks):
            # Generate a random substring for the ID
            random_suffix = uuid.uuid4().hex[:8]  # Using first 8 chars of a UUID
            # Create the unique ID for the chunk
            chunk_id = f"{document_common_id}_chunk_{i}_{random_suffix}"
            
            print(f"Adding chunk {i+1} with ID: {chunk_id}")
            add(ids=[chunk_id], docs=[chunk], collection_name="jds")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
