from pathlib import Path

from huggingface_hub import InferenceClient
import ollama
from pymongo import MongoClient, ReturnDocument
from pembot.TextEmbedder.mongodb_embedder import process_document_and_embed
from pembot.TextEmbedder.mongodb_index_creator import create_vector_index
import os
from enum import Enum
from datetime import datetime

class Category(Enum):
    WEBDEV     = "webdev"
    ANDROID    = "android"
    PHYSICS    = "physics"
    SPACE      = "space"
    PHILOSOPHY = "philosophy"
    CULTURE    = "culture"
    HISTORY    = "history"
    MATHS      = "maths"
    SOCIOLOGY  = "sociology"
    CODING     = "coding"

def get_title(input_text: str) -> str:
    #   1) splits `input_text` on "\n" → an array
    #   2) takes the [0]th element → the first line
    #   3) trims that element

    s= input_text.split('\n')[0].strip()

    if s.startswith("#"):
        return s[1:].strip()
    return s.strip()


def embed_all_docs(database, llm_client, inference_client, chunk_size, embedding_model, embeddings_collection_name, index_name, docs_collection):
    for doc in docs_collection.find({}, {"_id": 1, "content": 1}):

        print("on doc: ", doc)

        docs_embed= process_document_and_embed(db_client= database, llm_client= llm_client,
            inference_client= inference_client, file_path= Path(), chunk_size= chunk_size,
            embedding_model= embedding_model, embeddings_collection_name= embeddings_collection_name,
            use_custom_id= str(doc["_id"]), use_custom_input= doc["content"])

        doc_embedding_length= len(docs_embed[-1]["embedding"])

        # ensuring
        create_vector_index(database[embeddings_collection_name], index_name, num_dimensions= doc_embedding_length)

def update_from_local_fs_using_title(database, llm_client, inference_client, docs_dir, chunk_size, embedding_model, embeddings_collection_name, index_name, docs_collection):

    for docfile in docs_dir.iterdir():
        # for each blog in blog dir, update it if its not alr there in blogs coll,
        # that that created/updated doc's id
        # and update the embeddings in blog_embeddings collection with docId as that id
        # the docId of the embedding doc must be id of the corresponding blog
        if docfile.is_file :
            # TODO: create/update doc and get doc._id, pass that as the optional string param "use_custom_id"

            input_text= ""
            with open(str(docfile), "r") as md_file:
                input_text = md_file.read()

            categories= list([[i + 1, x, x.value] for i, x in enumerate(Category)])
            category= "culture" # default
            while True:
                category_index = input("Enter category no. for {}...\n{}\n: ".format(
                                    input_text[:300],
                                    '\n'.join(map(lambda x: str(x[0]) + ". " + str(x[2]), categories))
                                ))
                if category_index.isdigit():
                    category= categories[int(category_index) - 1][2]
                    break
                else:
                    print("pls enter a number.")

            set_on_insert= {
                "name": os.getenv("PROFILE_NAME"),
                "email": os.getenv("PROFILE_EMAIL"),
                "category": category,
                "dateAdded": datetime.now(),
                "views": 0,
                "likes": 0
            }

            title= get_title(input_text)

            updated= docs_collection.find_one_and_update(
                {'title': title},
                {
                    '$set': {'content': input_text},
                    '$setOnInsert': set_on_insert
                },
                upsert=True,
                return_document=ReturnDocument.AFTER
            )

            docs_embed= process_document_and_embed(db_client= database, llm_client= llm_client,
                inference_client= inference_client, file_path= docfile, chunk_size= chunk_size,
                embedding_model= embedding_model, embeddings_collection_name= embeddings_collection_name,
                use_custom_id= str(updated["_id"]))
            doc_embedding_length= len(docs_embed[-1]["embedding"])

            # ensuring
            create_vector_index(database[embeddings_collection_name], index_name, num_dimensions= doc_embedding_length)
            print("in the db now!")

if __name__ == "__main__":

    mongodb_uri= os.environ['MONGODB_PEM']
    mc = MongoClient(mongodb_uri)

    llm_client= ollama.Client()

    #### FOR USING JINA INSTEAD OF HUGGINGFACE SDK, REPLACE WITH THE InferenceClient TOP IMPORT
    # from pembot.utils.inference_client import InferenceClient
    # JINA_API_KEY= os.environ['JINA_API_KEY']
    # inference_client= InferenceClient(
    #     provider="Jina AI",
    #     api_key= JINA_API_KEY,
    # )

    inference_client= InferenceClient(
        provider="hf-inference",
        api_key= os.environ["HF_TOKEN"],
    )

    mc.admin.command('ping')
    print("ping test ok")
    database = mc["pem"]
    print("dbs and cols loaded")

    embeddings_collection_name: str= "blog_chunks"
    docs_collection_name: str= "blogs"
    docs_collection= database[docs_collection_name]

    # nerfed, but provided by hf serverless inference: BAAI/bge-small-en-v1.5
    # Worth mentioning:
    # jinaai/jina-embeddings-v3
    # BAAI/bge-base-en-v1.5
    # nomic-ai/nomic-embed-text-v1.5
    # embedding_model: str= 'BAAI/bge-base-en-v1.5'
    embedding_model: str= 'gemini-embedding-exp-03-07'

    index_name: str=  "gemini_vectors"

    # the number depends on the amount of chunks that will go into LLMs in the end
    total_input_chars_allowed= 2_50_000 # we got 63k tokens => ~2.5 lac characters
    no_of_chunks_to_be_sent = 2 #len(required_fields)
    chunk_size= int( total_input_chars_allowed / no_of_chunks_to_be_sent)

    # curdir <- project dir <- dev -> (~) -> Documents -> notes -> blogs
    docs_dir = Path.cwd().parent.parent.parent / "Documents" / "notes" / "Obsidian" / "blogs"

    embed_all_docs(database, llm_client, inference_client, chunk_size, embedding_model, embeddings_collection_name, index_name, docs_collection)
    # update_from_local_fs_using_title(database, llm_client, inference_client, docs_dir, chunk_size, embedding_model, embeddings_collection_name, index_name, docs_collection)
