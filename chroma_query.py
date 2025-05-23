import ollama
from TextEmbedder.chromadb_upload import querydb, add
import re
from google import genai
import os
from pembot.schema.structure import FineStructure


def gemini_query(rag_prompt, model_name):

    client = genai.Client(api_key= os.environ.get("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model= model_name,
        contents= rag_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": FineStructure,
        },
    )

    # Use instantiated objects ..
    outobject = response.parsed
    print(outobject)

    # Use the response as a JSON string.
    return response.text



def rag_query_llm(user_query: str, model_name: str = "qwen3:4b", ollama_base_url: str = "http://localhost:11434", no_of_fields= 4):
    """
    Performs a RAG (Retrieval Augmented Generation) query using a Hugging Face
    embedding model, ChromaDB for retrieval, and a local Ollama model for generation.

    Args:
        user_query (str): The user's query.
        model_name (str): The name of the Ollama model to use (e.g., "llama2", "mistral").
        ollama_base_url (str): The base URL for the local Ollama instance.

    Returns:
        str: The generated response from the Ollama model.
    """

    models = ollama.list()
    found= False

    for model in models.models:
        print(model.model)
        if model.model == model_name:
            found= True
            break


    # print("Step 2: Fetching relevant chunks from ChromaDB...")
    # You'll need to define your collection name
    relevant_chunks = querydb("jds", user_query, n_results=no_of_fields) # Fetch top 3 relevant chunks

    if not relevant_chunks:
        # print("No relevant chunks found in ChromaDB. Generating response without context.")
        context = "No relevant context available."
    else:
        # print(f"Found {len(relevant_chunks)} relevant chunks.")
        # Concatenate relevant chunks into a single context string
        context = "\n\n".join([chunk['document'] for chunk in relevant_chunks])

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
    if found:
        try:
            # You can use ollama.chat or ollama.generate depending on your model and preference
            # ollama.chat is generally preferred for conversational models.
            response = ollama.chat(
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
    elif 'gemini' in model_name:
        return gemini_query(rag_prompt, model_name= model_name)
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
    ctx= [
            "HCLTech is a global technology company.",
            "They specialize in AI engineering and cloud solutions.",
            "The company was founded in 1976.",
            "AI engineering focuses on developing intelligent systems.",
            "Cloud solutions help businesses scale their infrastructure."
    ]

    add(
        collection_name= 'jds',
        docs=ctx,
        ids=["doc1_chunk1", "doc1_chunk2", "doc1_chunk3", "doc1_chunk4", "doc1_chunk5"]
    )

    print(ctx)

    query = "What does HCLTech specialize in?"
    model_name= "qwen3:4b"
    print(query)
    response = rag_query_llm(user_query= query, model_name=model_name)
    # print("\n--- RAG Response ---")
    print(remove_bs(response))

    query2 = "When was HCLTech founded?"
    print(query2)
    response2 = rag_query_llm(query2, model_name= model_name)
    print(remove_bs(response2))

    query3 = "Tell me about their marketing strategies."
    response3 = rag_query_llm(query3, model_name= model_name)
    print(query3)
    # print("\n--- RAG Response 3 (Expected to be without context) ---")
    print(remove_bs(response3))
