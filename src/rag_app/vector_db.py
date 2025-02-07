import os
import chromadb
from chromadb.utils import embedding_functions

# Configure the path to anticipate eror in ChromaDB
CHROMA_DB_PATH = os.path.abspath("../data")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Default model for embedding
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def get_collection_names():
    """return collection name in ChromaDB"""
    return chroma_client.list_collections()


def register_collection(collection_name):
    """ Assign collection to COLLECTIONS.txt"""
    with open("COLLECTIONS.txt", "a") as f:
        f.write(collection_name + "\n")

def create_vector_db(docs, model_name=DEFAULT_EMBEDDING_MODEL, collection_name="default_collection"):
    """Make and load vector database"""
    # Check if collections exist
    existing_collections = chroma_client.list_collections() 

    if collection_name in existing_collections:
                collection = chroma_client.get_collection(name=collection_name)
    else:
        # make a embedding fuction without device
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function)
            register_collection(collection_name)

    # adding document to collection
    num_ids = collection.count()
    num_docs = len(docs)    
    collection.add(
        documents=[doc.page_content for doc in docs],
        ids=[f'id_{i}' for i in range(num_ids, num_ids + num_docs)],
        metadatas=[doc.metadata for doc in docs]
    )

    return collection

def load_local_db(collection_name):
    """ Load collection from local database """
    collection = chroma_client.get_collection(name=collection_name)
    return collection

def get_collection_data(collection_name: str):
    """
    return selected of collection from chromaDB
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
        documents = collection.get()
        return {"collection_name": collection_name, "documents": documents}
    except Exception as e:
        return {"error": str(e)}
