from pembot.TextEmbedder.chromadb_upload import add, clear_collection
import uuid
from pembot.AnyToText.convertor import chunk_text
import re
import random
from pathlib import Path


def make_it_an_id(file_name):
    """
    Input: A file name mixed with spaces, periods, etc.
    Output: '_' separated smallcase alphabetic id, with random filler if less than 5 chars
    """
    # 1. Convert to lowercase
    file_name = file_name.lower()

    # 2. Replace non-alphanumeric characters (except periods and spaces) with underscores
    # Keep periods for now to split by later, and spaces for initial conversion
    cleaned_name = re.sub(r'[^a-z0-9\s\.]', '_', file_name)
    
    # 3. Replace spaces and periods with underscores
    cleaned_name = re.sub(r'[\s\.]+', '_', cleaned_name)

    # 4. Remove leading/trailing underscores and multiple consecutive underscores
    cleaned_name = re.sub(r'_{2,}', '_', cleaned_name).strip('_')

    # Ensure it only contains alphabetic characters (after previous cleaning)
    # If the file_name was something like "123.pdf", this step ensures we only keep alphabetic parts.
    # We will filter out non-alphabetic parts after initial cleaning to retain some structure.
    # Let's refine this to ensure we only keep alphabetic parts before padding.
    alphabetic_parts = re.findall(r'[a-z]+', cleaned_name)
    
    # Join alphabetic parts with underscores
    result_id = '_'.join(alphabetic_parts)

    # 5. Add random filler if less than 5 chars
    if len(result_id) < 5:
        # Generate random lowercase alphabetic characters
        filler_length = 5 - len(result_id)
        random_filler = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(filler_length))
        
        # Append filler with an underscore if result_id is not empty
        if result_id:
            result_id += '_' + random_filler
        else: # If the result_id is empty (e.g., from "123.txt" or "$.*"), just use the filler
            result_id = random_filler
            
    return result_id


def upload_textfile(file_path, collection_name= 'jds'):

    try:
        with open(str(file_path), "r") as md_file:
            full_document_text = md_file.read()

        # Chunk the document
        chunks = chunk_text(full_document_text, chunk_size=1000, overlap_size=100) # Adjust chunk_size and overlap_size as needed

        
        #### IF WE ARE TRYING TO SEARCH INSIDE EACH DOCUMENT
        clear_collection(collection_name)

        ### IF GLOBAL SEARCH IS REQUIRED INSTEAD OF LOCAL, REMOVE THE ABOVE LINE

        # Add chunks to ChromaDB
        for i, chunk in enumerate(chunks):
            # Generate a random substring for the ID
            random_suffix = uuid.uuid4().hex[:8]  # Using first 8 chars of a UUID
            # Create the unique ID for the chunk
            chunk_id = f"{make_it_an_id(file_path.name)}_chunk_{i}_{random_suffix}"
            
            print(f"Adding chunk {i+1} with ID: {chunk_id}")
            # print(chunk)

            add(ids=[chunk_id], docs=[chunk], collection_name= collection_name)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    print("upload textfile and make it an id function")
