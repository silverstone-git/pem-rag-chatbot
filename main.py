from pathlib import Path

from huggingface_hub import InferenceClient
from huggingface_hub.inference._providers import PROVIDER_T
import ollama
from pymongo import MongoClient
from pembot.AnyToText.convertor import Convertor
from pembot.TextEmbedder.mongodb_embedder import process_document_and_embed
from pembot.query import rag_query_llm, remove_bs
import os
import json
from pembot.utils.string_tools import make_it_an_id
import pickle
from sys import argv

required_fields_path= ""
required_fields= None


def make_query(required_fields: list[tuple[str, str, str, str]]):
    """
    Makes a query to get json in the form of required fields
    """
    # Construct the part of the prompt that defines the desired JSON structure
    json_structure_definition = "{\n"
    for i, field in enumerate(required_fields):
        field_name, field_type, field_description, default_value = field
        json_structure_definition += f'  "{field_name}": "({field_type}) <{field_description}, default: {default_value}>"'
        if i < len(required_fields) - 1:
            json_structure_definition += ",\n"
        else:
            json_structure_definition += "\n"
    json_structure_definition += "}"

    # Construct the full query
    query = (
        "Extract the following information from the above provided context and return it as a JSON object. "
        "Ensure the output strictly conforms to the JSON format. "
        "Use the default values if the information is not found in the text.\n"
        "The required JSON structure is:\n"
        f"{json_structure_definition}\n\n"
        "JSON Output:"
    )
    return query

def save_to_json_file(llm_output: str, filepath: Path):
    """
    Takes JSON string and puts it in a .json file
    """
    try:
        # Ensure the directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Attempt to parse the string to validate it's JSON
        # and to get a nicely formatted string for the file.
        json_data = json.loads(llm_output)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved JSON to {filepath}")
        return json_data
    except json.JSONDecodeError:
        print(f"Error: LLM output is not valid JSON. Could not save to {filepath}")
        print("LLM Output was:\n", llm_output)
        # Optionally, save the raw invalid output for debugging
        # raw_output_path = filepath.with_suffix('.raw_llm_output.txt')
        # with open(raw_output_path, 'w', encoding='utf-8') as f:
        #     f.write(llm_output)
        # print(f"Raw LLM output saved to {raw_output_path}")
    except IOError as e:
        print(f"Error saving file to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in save_to_json_file: {e}")

def make_document_summarization_and_embeddings(db_client, llm_client, inference_client, docs_dir: Path, text_out_dir: Path, required_fields: list[tuple[str, str, str, str]], chunk_size: int = 600, embedding_model: str= 'nomic-embed-text:v1.5', llm_provider_name: PROVIDER_T= "novita", model_name= "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", embeddings_collection: str= "doc_chunks", index_name= "test_search"):
    # give required output fields
    # take the documents
    # convert to text
    # upload to chromadb
    # query the required fields

    for docfile in docs_dir.iterdir():

        file_root= os.path.splitext(docfile.name)[0]
        expected_json= text_out_dir / 'json' / (file_root + '.json')
        document_id= make_it_an_id(file_root)

        if docfile.is_file and not (expected_json).exists():

            expected_markdown= text_out_dir / (file_root + '.md')
            if not (expected_markdown).exists():
                converted= Convertor(docfile, text_out_dir)
            print("markdown made.", text_out_dir)

            # the case that convertor() made the json
            if expected_json.exists():
                continue

            # text files will be chunked and stored in separate persistent vector collections
            process_document_and_embed(db_client, llm_client, inference_client, expected_markdown, chunk_size= chunk_size, embedding_model= embedding_model, embeddings_collection_name= embeddings_collection)
            print("its in the db now")

            query= make_query(required_fields)
            print("full query is: ")
            print(query)
            filename_string= file_root + '.json'
            required_fields_descriptions= list(map(lambda x: x[1], required_fields))
            llm_output= rag_query_llm(db_client, llm_client, inference_client, query, document_id, required_fields_descriptions, no_of_fields= len(required_fields), llm_provider_name= llm_provider_name, model_name= model_name, embedding_model= embedding_model, embeddings_collection= embeddings_collection, index_name= index_name)

            # llm_output= rag_query_llm(query, no_of_fields= len(required_fields))
            jsonstr= remove_bs(llm_output)

            save_to_json_file(jsonstr, text_out_dir / 'json' / filename_string)


def upload_summaries(json_dir: Path, docs_collection):

    for json_path in json_dir.iterdir():

        base_name, _ = os.path.splitext(json_path.name)
        corresponding_text_file= json_dir.parent / (base_name + ".md")
        document_name_id= make_it_an_id(base_name)

        with open(str(json_path)) as json_file:
            json_data= json.load(json_file)

            print("pushing doc: ", corresponding_text_file, json_path)
            result = docs_collection.update_one(
                {"document_name_id": document_name_id}, # Filter by the content field
                {"$setOnInsert": {**json_data, "document_name_id": document_name_id}}, # Set these fields ONLY on insert
                upsert=True # Insert if no matching document is found
            )

            if result.upserted_id:
                print(f"New Document inserted with _id: {result.upserted_id}")
            else:
                print("Document with this docId found. Updated.")



def initit(db_client, llm_client, inference_client, chunk_size= 500, embedding_model= "BAAI/bge-en-icl", llm_provider_name: PROVIDER_T= "novita", model_name=  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", embeddings_collection: str= "doc_chunks", index_name= "test_search"):

    local_project_files_dir= Path.cwd().parent
    docs= local_project_files_dir / 'documents'
    text_out= local_project_files_dir / 'text-outputs'

    docs.mkdir(parents= True, exist_ok= True)
    text_out.mkdir(parents= True, exist_ok= True)

    make_document_summarization_and_embeddings(db_client, llm_client, inference_client, docs, text_out, required_fields, chunk_size= chunk_size, embedding_model= embedding_model, llm_provider_name= llm_provider_name, model_name= model_name, embeddings_collection= embeddings_collection, index_name= index_name)

    return text_out


if __name__ == "__main__":

    mongodb_uri= os.environ['MONGODB_PEMBOT']
    mc = MongoClient(mongodb_uri)

    llm_client= ollama.Client()

    #### FOR USING JINA INSTEAD OF HUGGINGFACE SDK, REPLACE WITH THE InferenceClient TOP IMPORT
    # from pembot.utils.inference_client import InferenceClient
    # JINA_API_KEY= os.environ['JINA_API_KEY']
    # inference_client= InferenceClient(
    #     provider="Jina AI",
    #     api_key= JINA_API_KEY,
    # )
    #

    try:
        if len(argv) > 1:
            print(f"First argument: {argv[1]}")
            required_fields_path= argv[1]
            with open(required_fields_path, "rb") as rf:
                required_fields= pickle.load(rf)
    except Exception as e:
        print("error while getting required_fields pickle. Please pickle it and put it in project directory to continue\n", e)

    if required_fields is None:
        print("couldnt load required fields. please provide path to pickle in command line argument")
        exit()
    else:
        print(required_fields)


    inference_client= InferenceClient(
        provider="hf-inference",
        api_key= os.environ["HF_TOKEN"],
    )

    mc.admin.command('ping')
    print("ping test ok")
    database = mc["pembot"]
    print("dbs and cols loaded")

    embeddings_collection: str= "doc_chunks"

    # if you want to use LLM inference from a different provider than embeddings
    llm_provider_name: PROVIDER_T="nebius"

    # nerfed, but provided by hf serverless inference: BAAI/bge-small-en-v1.5
    # Worth mentioning:
    # jinaai/jina-embeddings-v3
    # BAAI/bge-base-en-v1.5
    # nomic-ai/nomic-embed-text-v1.5
    # embedding_model: str= 'BAAI/bge-base-en-v1.5'
    embedding_model: str= 'gemini-embedding-exp-03-07'
    # model_name: str= "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # model_name: str= "google/gemma-3-27b-it"
    model_name: str= "gemini-2.5-flash-preview-05-20"

    index_name: str=  "gemini_vectors"

    # output tokens are ~1000 at max
    # chunk_size= 1000 # this is for chhote mote models, gemma 1b or smth
    chunk_size= int(2_50_000 / len(required_fields)) # we got 63k tokens => ~2.5 lac characters


    #### REQUIRED_FIELDS:
    # an array of tuples:
    # (field name, field description, field type, default value)

    process_output_dir= initit(database, llm_client, inference_client, chunk_size= chunk_size, embedding_model= embedding_model, llm_provider_name= llm_provider_name, model_name= model_name, embeddings_collection= embeddings_collection, index_name= index_name)

    docs_collection= database["summary_docs"]
    upload_summaries(process_output_dir / 'json', docs_collection)
