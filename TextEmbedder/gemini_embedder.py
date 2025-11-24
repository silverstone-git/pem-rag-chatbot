from google.genai import types
import requests
import json
import os
from google import genai


def get_embedding(client, text):

    result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents= text,
        config= types.EmbedContentConfig(task_type= "RETRIEVAL_DOCUMENT"),
    )

    if result:
        return result.embeddings[0].values
    else:
        print(f"Error: {result}")
        #raise ValueError("Error communicating with the gemini embedding API")


if __name__ == "__main__":

    api_key = os.environ['GEMINI_API_KEY']
    genai_client= genai.Client(api_key= api_key)
    print(get_embedding(genai_client, "send bobs"))
