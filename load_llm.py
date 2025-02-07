import requests
from src.rag_app.config import llm_config

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # URL yang benar

def load_groq_api_model(prompt):
    headers = {
        "Authorization": f"Bearer {''}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "mixtral-8x7b-32768",  # atau "llama2-70b-4096"
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": llm_config["max_tokens"],
        "temperature": llm_config["temperature"],
    }

    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error calling Groq API: {str(e)}"