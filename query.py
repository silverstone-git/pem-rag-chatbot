from os import environ
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionOutputMessage
from huggingface_hub.inference._providers import PROVIDER_T
import ollama
import re

from pydantic_core.core_schema import ErrorType
from pembot.TextEmbedder.mongodb_embedder import search_within_document
import numpy as np
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
import time

from pembot.TextEmbedder.mongodb_index_creator import create_vector_index

def external_llm(rag_prompt, model_name, llm_provider_name: PROVIDER_T= "novita", inference_client = None) -> str:

    # Here, one can change the provider of the inference LLM if
    # for embedding we are using one which doesnt have our LLM available
    # or, is costly, so we choose different, just here in the function header, or from the main()

    if not inference_client:
        inference_client= InferenceClient(
                # "nebius" "novita" "hyperbolic"
                provider= llm_provider_name,
                api_key= environ["HF_TOKEN"]
            )

    completion= inference_client.chat.completions.create(
            model= model_name,
            messages= [
                {"role": "user", "content": rag_prompt}
            ]
        )
    response_message: ChatCompletionOutputMessage= completion.choices[0].message

    if response_message.content:
        return response_message.content
    else:
        return '{}'


def multi_embedding_average(llm_client, inference_client, descriptions, model= "BAAI/bge-en-icl", embed_locally= False):

    description_embeddings = []
    for desc in descriptions:
        try:
            if 'gemini' in model:
                client = genai.Client(api_key= environ['GEMINI_API_KEY'])
                result = client.models.embed_content(
                        model= model,
                        contents= desc,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                if result is not None and result.embeddings:
                    description_embeddings.append(result.embeddings[0].values)
                else:
                    raise ValueError("Gemini not givingz embeddingzzz")
            elif embed_locally:
                response = llm_client.embeddings(model=model, prompt=desc)
                description_embeddings.append(response['embedding'])

            else:
                response = inference_client.feature_extraction(desc, model=model)
                description_embeddings.append(response)

        except Exception as e:
            print(f"Error generating embedding for description '{desc}': {e}")
            # Decide how to handle errors: skip, raise, or use a placeholder
            # continue
            raise e
        time.sleep(1)

    if not description_embeddings:
        print("No embeddings could be generated for the descriptions. Aborting search.")
        return []

    # Aggregate embeddings: A simple approach is to average them.
    # This creates a single query vector that represents the combined meaning.
    return np.mean(description_embeddings, axis=0).tolist()



def rag_query_llm(db_client, llm_client, inference_client, user_query: str, document_id: str, required_fields_descriptions: list[str], model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", ollama_base_url: str = "http://localhost:11434", no_of_fields= 4, embedding_model= "BAAI/bge-en-icl", llm_provider_name: PROVIDER_T= "novita", index_name: str= "test_search", embeddings_collection= "doc_chunks", document_belongs_to_a_type= ""):
    """
    Performs a RAG (Retrieval Augmented Generation) query using a Hugging Face
    embedding model, ChromaDB for retrieval, and a local Ollama model for generation.

    Args:
        db_client: The vector DB client
        user_query (str): The user's query.
        required_fields_descriptions: The required fields which are to be queried from context
        model_name (str): The name of the Ollama model to use (e.g., "llama2", "mistral").
        no_of_fields (str): number of vectors which are to be retrieved from DB

    Returns:
        str: The generated response from the Ollama model.
    """

    embed_locally= False
    found= False
    try:
        models = llm_client.list()
        for model in models.models:
            # print(model.model)
            if model.model == model_name:
                found= True
            if model.model == embedding_model:
                embed_locally= True
    except AttributeError as ae:
        print("cant find ollama", ae)
        print("continuing with other models")
    except Exception as e:
        print("unhandled error: ", e)
        raise e



    aggregate_query_embedding= multi_embedding_average(llm_client, inference_client, required_fields_descriptions, model= embedding_model, embed_locally= embed_locally)
    print("Aggregate query embedding generated. length: ", len(aggregate_query_embedding))

    create_vector_index(db_client[embeddings_collection], index_name, num_dimensions= len(aggregate_query_embedding), document_belongs_to_a_type= document_belongs_to_a_type)

    # check the order of args
    relevant_chunks= search_within_document(db_client, aggregate_query_embedding, document_id, limit= no_of_fields, index_name= index_name, embeddings_collection_name= embeddings_collection, document_belongs_to_a_type= document_belongs_to_a_type)
    relevant_chunks= list(map(lambda x: x['chunk_text'], relevant_chunks))

    if not relevant_chunks:
        context = "No relevant context available."
    else:
        # print(f"Found {len(relevant_chunks)} relevant chunks.")
        # Concatenate relevant chunks into a single context string
        context = "\n\n".join(relevant_chunks)

    # Construct the RAG prompt
    rag_prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context:
    {context}

    Question: {user_query}
    """

    # print("Step 3: Calling Ollama model with RAG prompt...")
    # print("final prompt: ")
    # print(rag_prompt)
    if 'gemini' in model_name:

        client = genai.Client(api_key= environ['GEMINI_API_KEY'])
        response = client.models.generate_content(
            model= model_name,
            contents= rag_prompt,
        )
        return response.text

    elif found:
        try:
            # You can use ollama.chat or ollama.generate depending on your model and preference
            # ollama.chat is generally preferred for conversational models.
            response = llm_client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': rag_prompt}],
                options={"base_url": ollama_base_url} # Ensure the base URL is set
            )
            return response['message']['content']
        except ollama.ResponseError as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: Could not get a response from Ollama. Please check if Ollama is running and the model '{model_name}' is pulled."
        except Exception as e:
            print(f"An unexpected error occurred while calling Ollama: {e}")
            return "An unexpected error occurred."
    elif 'qwen' in model_name or 'gemma' in model_name or 'Qwen' in model_name or 'deepseek' in model_name:
        return external_llm(rag_prompt, model_name= model_name, llm_provider_name= llm_provider_name)
    else:
        return '{}'



def remove_bs(text):
    """
    Removes everything between <think></think> tags and any text outside of JSON curly brackets.

    Args:
        text (str): The input string.

    Returns:
        str: The string with text between <think></think> tags removed and only the
             content within the outermost JSON curly brackets.
             Returns an empty string if no valid JSON is found.
    """
    # 1. Remove <think></think> tags
    think_pattern = r'<think>.*?</think>'
    text_without_think = re.sub(think_pattern, '', text, flags=re.DOTALL)

    # 2. Extract JSON content
    # This regex looks for the first opening curly brace and the last closing curly brace.
    # It assumes the JSON structure is well-formed within the string.
    json_match = re.search(r'\{(.*)\}', text_without_think, re.DOTALL)

    if json_match:
        json_content_str = "{" + json_match.group(1) + "}"
        return json_content_str
    else:
        return ""


if __name__ == "__main__":
    print("hemlo worls")
