import requests
import json
import os


def get_embedding(text):
    api_key = os.environ['GEMINI_API_KEY']
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"

    data = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{
                "text": text
            }]
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        resJ = response.json()
        return resJ['embedding']['values']
    else:
        print(f"Error: {response.status_code}, {response.text}")
        #raise ValueError("Error communicating with the gemini embedding API")


if __name__ == "__main__":
    print(get_embedding("send bobs"))
