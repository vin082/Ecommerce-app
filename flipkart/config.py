import os
from dotenv import load_dotenv

# hugging face gets automatically loaded 
load_dotenv()

class Config:
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    #GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RAG_MODEL = "groq:qwen/qwen3-32b"
    #RAG_MODEL = "groq:llama-3.1-8b-instant"
    #RETRIEVER_MODEL = "groq:llama-3.1-8b-instant"  # lightweight model for MultiQueryRetriever query expansion
    RETRIEVER_MODEL = "openai:gpt-4o-mini"
