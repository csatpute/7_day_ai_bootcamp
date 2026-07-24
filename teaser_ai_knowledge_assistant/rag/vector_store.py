import hashlib
from typing import Any

import chromadb

from config import CHROMA_PATH
from services.llm_service import embed_texts
from utils.pdf_utils import TextChunk


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )


def make_collection_name(
    filename: str,
    file_bytes: bytes,
) -> str:
    """Create a unique collection name for one PDF."""

    digest = hashlib.sha256(
        filename.encode("utf-8") + file_bytes
    ).hexdigest()[:24]

    return f"document_{digest}"


def index_chunks(
    collection_name: str,
    chunks: list[TextChunk],
    source_name: str,
    batch_size: int = 32,
) -> int:
    """Add document chunks and embeddings to ChromaDB."""

    if not chunks:
        raise ValueError(
            "There are no chunks to index."
        )

    client = _client()

    try:
        client.delete_collection(
            name=collection_name
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        documents = [
            item["text"]
            for item in batch
        ]

        embeddings = embed_texts(documents)

        ids = [
            f"{collection_name}_{item['chunk']}"
            for item in batch
        ]

        metadatas = [
            {
                "source": source_name,
                "page": item["page"],
                "chunk": item["chunk"],
            }
            for item in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection.count()


def retrieve_chunks(
    collection_name: str,
    question: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Find the chunks most closely related to a question."""

    if not question.strip():
        return []

    client = _client()

    collection = client.get_collection(
        name=collection_name
    )

    count = collection.count()

    if count == 0:
        return []

    query_embedding = embed_texts(
        [question]
    )[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result.get(
        "documents",
        [[]],
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]],
    )[0]

    distances = result.get(
        "distances",
        [[]],
    )[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": metadata.get(
                    "page",
                    "?",
                ),
                "chunk": metadata.get(
                    "chunk",
                    "?",
                ),
                "distance": distance,
            }
        )

    return retrieved


def format_retrieved_context(
    retrieved: list[dict[str, Any]],
) -> str:
    """Prepare retrieved chunks for the LLM prompt."""

    sections = []

    for index, item in enumerate(
        retrieved,
        start=1,
    ):
        sections.append(
            f"[Source {index}: "
            f"{item['source']}, "
            f"page {item['page']}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(sections)