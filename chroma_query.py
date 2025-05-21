import ollama
from TextEmbedder.chromadb_upload import querydb, add
import re

def rag_query_ollama(user_query: str, ollama_model_name: str = "qwen3:4b", ollama_base_url: str = "http://localhost:11434"):
    """
    Performs a RAG (Retrieval Augmented Generation) query using a Hugging Face
    embedding model, ChromaDB for retrieval, and a local Ollama model for generation.

    Args:
        user_query (str): The user's query.
        ollama_model_name (str): The name of the Ollama model to use (e.g., "llama2", "mistral").
        ollama_base_url (str): The base URL for the local Ollama instance.

    Returns:
        str: The generated response from the Ollama model.
    """

    # print("Step 2: Fetching relevant chunks from ChromaDB...")
    # You'll need to define your collection name
    relevant_chunks = querydb("jds", user_query, n_results=3) # Fetch top 3 relevant chunks

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

    Answer:
    """
    
    # print("Step 3: Calling Ollama model with RAG prompt...")
    # print("final prompt: ")
    # print(rag_prompt)
    try:
        # You can use ollama.chat or ollama.generate depending on your model and preference
        # ollama.chat is generally preferred for conversational models.
        response = ollama.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': rag_prompt}],
            options={"base_url": ollama_base_url} # Ensure the base URL is set
        )
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"Error calling Ollama API: {e}")
        return f"Error: Could not get a response from Ollama. Please check if Ollama is running and the model '{ollama_model_name}' is pulled."
    except Exception as e:
        print(f"An unexpected error occurred while calling Ollama: {e}")
        return "An unexpected error occurred."


def remove_think_tags(text):
    """
    Removes everything between <think></think> tags in a string.

    Args:
        text (str): The input string.

    Returns:
        str: The string with text between <think></think> tags removed.
    """
    pattern = r'<think>.*?</think>'  # Non-greedy match for text between tags
    return re.sub(pattern, '', text, flags=re.DOTALL)

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
    response = rag_query_ollama(user_query= query, ollama_model_name=model_name)
    # print("\n--- RAG Response ---")
    print(remove_think_tags(response))

    query2 = "When was HCLTech founded?"
    print(query2)
    response2 = rag_query_ollama(query2, ollama_model_name= model_name)
    print(remove_think_tags(response2))

    query3 = "Tell me about their marketing strategies."
    response3 = rag_query_ollama(query3, ollama_model_name= model_name)
    print(query3)
    # print("\n--- RAG Response 3 (Expected to be without context) ---")
    print(remove_think_tags(response3))
