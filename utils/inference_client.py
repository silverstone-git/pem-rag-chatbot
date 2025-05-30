import os
import requests
import json
from enum import Enum

# Define an Enum for providers if you plan to support more in the future
class PROVIDER_T(Enum):
    JINA = "Jina AI"
    # OLLAMA = "Ollama" # Example for future expansion

class InferenceClient:
    """
    A client for making inference calls to various AI model providers.
    Currently supports Jina AI for feature extraction (embeddings).
    """

    def __init__(self, provider: PROVIDER_T, api_key: str):
        """
        Initializes the InferenceClient.

        Args:
            provider: The AI model provider (e.g., PROVIDER_T.JINA).
            api_key: The API key for the specified provider.
        """
        self.provider = provider
        self.api_key = api_key
        self.base_url = self._get_base_url(provider)

    def _get_base_url(self, provider: PROVIDER_T) -> str:
        """
        Returns the base URL for the given provider's API.
        """
        if provider == PROVIDER_T.JINA:
            return "https://api.jina.ai/v1/embeddings"
        # Add more providers here as needed
        # elif provider == PROVIDER_T.OLLAMA:
        #     return "http://localhost:11434/api/embeddings"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def feature_extraction(self, prompt_texts: list[str], model: str = "jina-embeddings-v3") -> list[list[float]]:
        """
        Calls the Jina AI API to get embeddings (feature extraction) for input texts.

        Args:
            prompt_texts: A list of strings for which to generate embeddings.
            model: The name of the Jina AI embedding model to use.
                   Defaults to "jina-embeddings-v3".

        Returns:
            A list of embedding vectors (list of floats).

        Raises:
            Exception: If the API call fails or returns an error.
        """
        if self.provider != PROVIDER_T.JINA:
            raise NotImplementedError(f"Feature extraction not implemented for provider: {self.provider}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Jina API specific payload
        data = {
            "model": model,
            "input": prompt_texts,
            "task": "qa" # Recommended task for general purpose embeddings, can be "text-matching", "retrieval", etc.
                         # "qa" is often good for general question-answering/retrieval contexts.
        }

        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(data))
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_json = response.json()
            if "data" in response_json and isinstance(response_json["data"], list):
                # Extract just the embedding vectors
                embeddings = [item["embedding"] for item in response_json["data"]]
                return embeddings
            else:
                raise Exception(f"Unexpected API response format: {response_json}")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err} - {response.text}")
            raise
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
            raise
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"An unexpected request error occurred: {req_err}")
            raise
        except Exception as e:
            print(f"An error occurred during Jina AI API call: {e}")
            raise

# --- Example Usage ---
if __name__ == "__main__":
    try:
        JINA_API_KEY = os.environ['JINA_API_KEY']
    except KeyError:
        print("Error: JINA_API_KEY environment variable not set.")
        print("Please set it (e.g., export JINA_API_KEY='your_api_key') and try again.")
        exit(1)

    # Initialize the InferenceClient for Jina AI
    jina_client = InferenceClient(provider=PROVIDER_T.JINA, api_key=JINA_API_KEY)

    # Example input texts
    input_texts = [
        "Organic skincare for sensitive skin with aloe vera and chamomile.",
        "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille.",
        "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla.",
        "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています。"
    ]

    print("--- Calling Jina AI for Feature Extraction ---")
    try:
        embeddings = jina_client.feature_extraction(prompt_texts=input_texts)
        print(f"Successfully extracted {len(embeddings)} embeddings.")
        print(f"First embedding (first 5 values): {embeddings[0][:5]}...")
        print(f"Embedding dimension: {len(embeddings[0])}")

        # Example with a different model (if available and you want to specify)
        # embeddings_v2 = jina_client.feature_extraction(prompt_texts=input_texts, model="jina-embeddings-v2")
        # print(f"\nSuccessfully extracted {len(embeddings_v2)} embeddings with jina-embeddings-v2.")

    except Exception as e:
        print(f"Failed to extract features: {e}")
