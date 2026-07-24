import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "llama3.1",
)

EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
)

CHROMA_PATH = Path(
    os.getenv("CHROMA_PATH", "data/chroma")
)

RAG_TOP_K = int(
    os.getenv("RAG_TOP_K", "4")
)

CHROMA_PATH.mkdir(
    parents=True,
    exist_ok=True,
)