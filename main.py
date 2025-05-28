from pathlib import Path

import chromadb
from google import genai
from pembot.AnyToText.convertor import Convertor
from TextEmbedder.chroma_controller import upload_textfile
from pembot.chroma_query import rag_query_llm, remove_bs
import os
import json
from schema.structure import required_fields


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

def get_fieldvals(chroma_client, docs_dir: Path, text_out_dir: Path, required_fields: list[tuple[str, str, str, str]], chunk_size: int = 1000): 
    # give required output fields 
    # take the documents
    # convert to text
    # upload to chromadb
    # query the required fields

    for docfile in docs_dir.iterdir():

        file_root= os.path.splitext(docfile.name)[0]
        expected_json= text_out_dir / 'json' / (file_root + '.json')

        if docfile.is_file and not (expected_json).exists(): 

            expected_markdown= text_out_dir / (file_root + '.md')
            if not (expected_markdown).exists():
                converted= Convertor(docfile, text_out_dir)
            print("markdown made.", text_out_dir)

            # the case that convertor() made the json
            if expected_json.exists():
                continue

            upload_textfile(chroma_client, expected_markdown, collection_name= "jds", chunk_size= chunk_size)
            print("its in the db now")

            query= make_query(required_fields)
            print("full query is: ")
            print(query)
            filename_string= file_root + '.json'
            llm_output= rag_query_llm(chroma_client, genai_client, query, required_fields_descriptions= list(map(lambda x: x[1], required_fields)),  model_name= "gemini-2.5-flash-preview-05-20", no_of_fields= int(2 * len(required_fields)))

            # llm_output= rag_query_llm(query, no_of_fields= len(required_fields))
            jsonstr= remove_bs(llm_output)
            print(jsonstr)

            save_to_json_file(jsonstr, text_out_dir / 'json' / filename_string)



def initit(genai_client, chroma_client):

    local_project_files_dir= Path.cwd().parent
    docs= local_project_files_dir / 'documents'
    text_out= local_project_files_dir / 'text-outputs'

    docs.mkdir(parents= True, exist_ok= True)
    text_out.mkdir(parents= True, exist_ok= True)

    get_fieldvals(chroma_client, docs, text_out, required_fields, chunk_size= 600)


if __name__ == "__main__":

    api_key = os.environ['GEMINI_API_KEY']
    genai_client= genai.Client(api_key= api_key)

    chroma_client = chromadb.PersistentClient('/home/cyto/dev/pem-rag-chatbot/chroma')

    initit(genai_client, chroma_client)
