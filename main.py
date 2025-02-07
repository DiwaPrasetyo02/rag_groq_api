import argparse
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from pydantic import Field

from llama_index.llms.groq import Groq
from contextlib import asynccontextmanager
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile
from pydantic import BaseModel, Field

from src.rag_app.prompts import create_prompt
from src.rag_app.vector_db import create_vector_db, load_local_db, chroma_client, get_collection_names
from src.rag_app.load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
from src.rag_app.load_llm import load_groq_api_model
from src.rag_app.config import llm_config

def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}
text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "all-MiniLM-L6-v2"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["answer_to_query"] = load_groq_api_model
    # ml_models["answer_to_query"] = fake_output
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user upload a file and get the answer for the question using LLMs",
    lifespan=lifespan
)

@app.get("/")
def index():
    return {"message": "Hello World"}




@app.get("/init_llm")
def init_llama_llm(
    max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
    n_ctx: int = Query(4096, description="Token context window."),
    temperature: float = Query(0.7, description="Temperature for sampling. Higher values means more random samples.")
):
    llm_config["max_tokens"] = max_tokens
    llm_config["n_ctx"] = n_ctx
    llm_config["temperature"] = temperature

    ml_models["answer_to_query"] = load_groq_api_model

    return {"message": "LLM API initialized successfully", "llm_config": llm_config}


@app.post("/upload")
def upload_file(file: UploadFile = File(...), collection_name : Optional[str] = "test_collection"):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    if file.filename.endswith('.pdf'):
        data = load_split_pdf_file(f'../data/{file.filename}', text_splitter)
    elif file.filename.endswith('.html'):
        data = load_split_html_file(f'../data/{file.filename}', text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    db = create_vector_db(docs=data, model_name=vector_db_model_name, collection_name=collection_name)

    return {"message": f"Successfully uploaded {file.filename}",
            "num_splits" : len(data)}


@app.get("/collections")
def list_collections():
    """
    Endpoint to show list of collection that stored in ChromaDB
    """
    try:
        collections = chroma_client.list_collections()
        return {"collections": collections}
    except Exception as e:
        return {"error": str(e)}

@app.get("/collections/{collection_name}")
def get_collection_data_endpoint(collection_name: str):
    """
    Endpoint for showing collections
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        documents = collection.get()
        return {"collection_name": collection_name, "documents": documents}
    except Exception as e:
        return {"error": str(e)}


@app.get("/query")
def query(query: str, n_results: Optional[int] = 2, collection_name: Optional[str] = "test_collection"):
    collection_list = get_collection_names()
    
    if collection_name not in collection_list:
        return {"message": f"Collection {collection_name} not found",
                "available_collections": collection_list}

    collection = load_local_db(collection_name)
    results = collection.query(query_texts=[query], n_results=n_results)
    
    # Create prompt with context
    prompt = create_prompt(query, results)
    
    # Get LLM response
    llm_output = ml_models["answer_to_query"](prompt)

    return {
        "query": query,
        "relevant_docs": results,
        "llm_response": llm_output
    }


if __name__ == "__main__":
    pass

