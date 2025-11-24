from os import environ
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionOutputMessage
from huggingface_hub.inference._providers import PROVIDER_T
import ollama
import re
from smolagents import InferenceClientModel, ToolCallingAgent, ActionStep, TaskStep
from smolagents.default_tools import VisitWebpageTool
from pymongo import MongoClient
from typing import Callable, Dict, Any, Optional, List
import uuid
from datetime import datetime
from smolagents.monitoring import Timing


from pembot.search import brave_search_tool
from pembot.TextEmbedder.mongodb_embedder import search_within_document
import numpy as np
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
import time
from datetime import timezone

init_timing= {
    "start_time": 0.0,
    "end_time": 0.0,
    "duration": 0.0,
}

mongodb_uri= environ['MONGODB_SCHEMER']
mc = MongoClient(mongodb_uri)
db = mc["schemerdb"]
collection = db["chat_history"]  # Collection name

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



def rag_query_llm(db_client, llm_client, inference_client,
    user_query: str, document_id: str, required_fields_descriptions: list[str],
    model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    ollama_base_url: str = "http://localhost:11434", no_of_fields= 4,
    embedding_model= "BAAI/bge-en-icl", llm_provider_name: PROVIDER_T= "novita",
    index_name: str= "test_search", embeddings_collection= "doc_chunks",
    document_belongs_to_a_type= "", prompt_prefix= ""):
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
    {prompt_prefix}
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


def smolquery(message: str, external_tools: list[Callable] = [], chat_id: str | None = None, allow_web_search= True) -> Dict[str, Any]:
    """
    Run agent with chat history support.

    Args:
        message: User's message
        external_tools: List of external tools to use
        chat_id: Optional chat ID for continuing conversation
        allow_web_search: Boolean to decide whether to include brave tool to fetch search results
                            and the Visiting Web Page Tool in the agent's toolbox

    Returns:
        Dictionary containing response and chat_id
    """
    alltools = []

    if allow_web_search:
        alltools.extend([
            brave_search_tool,
            VisitWebpageTool(),
        ])

    alltools.extend(external_tools)

    model = InferenceClientModel(
        token= environ["HF_TOKEN"],
        # model_id= "HuggingFaceTB/SmolLM3-3B"
        model_id= "deepseek-ai/DeepSeek-R1-0528"
    )

    agent = ToolCallingAgent(tools=alltools, model=model, add_base_tools=False)

    # Handle chat history
    if chat_id:
        # Load existing conversation
        chat_doc = collection.find_one({"_id": chat_id})
        if chat_doc:
            # Restore agent memory from database
            restore_agent_memory(agent, chat_doc["messages"])
        else:
            # Chat ID provided but not found, create new one
            chat_id = str(uuid.uuid4())
    else:
        # Create new chat
        chat_id = str(uuid.uuid4())

    # Run the agent
    response = agent.run(message, reset= False)

    # Extract the final answer from the response
    final_answer = extract_final_answer(response)

    # Save conversation to database
    save_chat_history(chat_id, agent, message, final_answer)

    return {
        "response": final_answer,
        "chat_id": chat_id
    }

def extract_final_answer(response: Any) -> str:
    """
    Extract the final answer from various response types.

    Args:
        response: Response from agent.run()

    Returns:
        Final answer as string
    """
    # Handle RunResult object
    if hasattr(response, 'final_answer'):
        return str(response.final_answer)

    # Handle direct string response
    if isinstance(response, str):
        return response

    # Handle generator response
    if hasattr(response, '__iter__') and not isinstance(response, (str, bytes)):
        final_step = None
        for step in response:
            final_step = step
            # Look for FinalAnswerStep
            if hasattr(step, 'final_answer'):
                return str(step.final_answer)

        # If no final answer found, return last step as string
        if final_step is not None:
            return str(final_step)

    # Fallback to string conversion
    return str(response)

def restore_agent_memory(agent: ToolCallingAgent, messages: List[Dict[str, Any]]) -> None:
    """
    Restore agent memory from stored messages.

    Args:
        agent: The agent instance
        messages: List of stored messages
    """
    for msg in messages:
        if msg["type"] == "task":
            # Add task step
            task_step = TaskStep(
                task=msg["content"],
                task_images=msg.get("images", [])
            )
            agent.memory.steps.append(task_step)
        elif msg["type"] == "action":
            # Add action step with only the required parameters
            # ActionStep objects are typically created during execution
            # and contain read-only information, so we create a minimal one
            action_saved_timing= msg.get("timing", init_timing)
            action_step = ActionStep(
                observations= msg.get("observations", ""),
                step_number=msg["step_number"],
                observations_images=msg.get("observations_images", []),
                timing=Timing(
                    start_time= action_saved_timing.get("start_time", 0.0),
                    end_time= action_saved_timing.get("end_time", 0.0)
                )
            )
            agent.memory.steps.append(action_step)

def save_chat_history(chat_id: str, agent: ToolCallingAgent, user_message: str, agent_response: str) -> None:
    """
    Save conversation history to MongoDB.

    Args:
        chat_id: Chat session ID
        agent: Agent instance with memory
        user_message: Latest user message
        agent_response: Agent's response
    """
    # Convert agent memory to serializable format
    messages = []

    for step in agent.memory.steps:
        if isinstance(step, TaskStep):
            messages.append({
                "type": "task",
                "content": step.task,
                "images": step.task_images if hasattr(step, 'task_images') else [],
                "timestamp": datetime.now(timezone.utc)
            })
        elif isinstance(step, ActionStep):
            msg = {
                "type": "action",
                "step_number": step.step_number,
                "observations_images": step.observations_images if hasattr(step, 'observations_images') else [],
                "timing": step.timing.dict() if hasattr(step, 'timing') else init_timing,
                "timestamp": datetime.now(timezone.utc)
            }

            # Store any additional attributes that might be accessible
            # Note: ActionStep attributes are typically read-only
            if hasattr(step, 'observations') and step.observations:
                msg["observations"] = str(step.observations)
            if hasattr(step, 'error') and step.error:
                msg["error"] = str(step.error)

            messages.append(msg)

    # Add the latest response
    messages.append({
        "type": "response",
        "content": agent_response,
        "timestamp": datetime.now(timezone.utc)
    })

    # Update or insert chat document
    collection.update_one(
        {"_id": chat_id},
        {
            "$set": {
                "messages": messages,
                "last_updated": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )

def get_chat_history(chat_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve chat history by ID.

    Args:
        chat_id: Chat session ID

    Returns:
        List of messages or None if not found
    """
    chat_doc = collection.find_one({"_id": chat_id})
    return chat_doc["messages"] if chat_doc else None

def delete_chat_history(chat_id: str) -> bool:
    """
    Delete chat history by ID.

    Args:
        chat_id: Chat session ID

    Returns:
        True if deleted, False if not found
    """
    result = collection.delete_one({"_id": chat_id})
    return result.deleted_count > 0

def list_chat_sessions() -> List[Dict[str, Any]]:
    """
    List all chat sessions with basic info.

    Returns:
        List of chat sessions with ID and last updated time
    """
    sessions = []
    for doc in collection.find({}, {"_id": 1, "last_updated": 1, "messages": {"$slice": 1}}):
        first_message = doc["messages"][0] if doc["messages"] else {}
        sessions.append({
            "chat_id": doc["_id"],
            "last_updated": doc.get("last_updated"),
            "first_message": first_message.get("content", "")[:100] + "..." if len(first_message.get("content", "")) > 100 else first_message.get("content", "")
        })
    return sessions


# # First message - creates new chat
# result1 = smolquery("Hello, what's the weather like?", [])
# print(f"Response: {result1['response']}")
# print(f"Chat ID: {result1['chat_id']}")

# # Second message - continues the conversation
# result2 = smolquery("Thanks, now tell me about Python programming", [], chat_id=result1['chat_id'])
# print(f"Response: {result2['response']}")
# print(f"Chat ID: {result2['chat_id']}")  # Should be the same as result1['chat_id']

# # Retrieve chat history
# history = get_chat_history(result1['chat_id'])
# print(f"Chat history length: {len(history) if history else 0}")



if __name__ == "__main__":
    print("hemlo worls")

    # result1 = smolquery("Did i tell you to do something regarding stocks before? What do you conclude?", allow_web_search= False, chat_id= "a52ab59e-d6d0-4089-a963-61e8876244e0")
    result1 = smolquery("How has NIFTY 50 been doing past 3 months?")
    print(f"Response: {result1['response']}")
    print(f"Chat ID: {result1['chat_id']}")

    # # Second message - continues the conversation
    result2 = smolquery("now tell me about other indices in the same country", chat_id=result1['chat_id'])
    print(f"Response: {result2['response']}")
    print(f"Chat ID: {result2['chat_id']}")  # Should be the same as result1['chat_id']

    # # Retrieve chat history
    history = get_chat_history(result1['chat_id'])
    print(f"Chat history length: {len(history) if history else 0}")
