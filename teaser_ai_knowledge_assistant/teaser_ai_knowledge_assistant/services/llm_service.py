from typing import Any

import ollama

from config import CHAT_MODEL, EMBEDDING_MODEL


DEFAULT_SYSTEM_PROMPT = (
    "You are a practical AI assistant. Give accurate, clear, "
    "well-structured answers. State uncertainty when appropriate."
)


def _read_message_content(response: Any) -> str:
    """Read text from an Ollama response."""

    if hasattr(response, "message"):
        return response.message.content

    return response["message"]["content"]


def generate_response(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.2,
) -> str:
    """Send one prompt to the configured Ollama model."""

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": temperature,
            },
        )

        return _read_message_content(response).strip()

    except Exception as exc:
        raise RuntimeError(
            "The LLM request failed. Confirm that Ollama is "
            f"running and that '{CHAT_MODEL}' is installed. "
            f"Original error: {exc}"
        ) from exc


def generate_chat_response(
    history: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Generate a response using the complete chat history."""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    messages.extend(history)

    try:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={
                "temperature": 0.3,
            },
        )

        return _read_message_content(response).strip()

    except Exception as exc:
        raise RuntimeError(
            "The chat request failed. Confirm that Ollama is "
            f"running and that '{CHAT_MODEL}' is installed. "
            f"Original error: {exc}"
        ) from exc


def embed_texts(
    texts: list[str],
) -> list[list[float]]:
    """Create embeddings for a collection of strings."""

    if not texts:
        return []

    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        if hasattr(response, "embeddings"):
            return response.embeddings

        return response["embeddings"]

    except Exception as exc:
        raise RuntimeError(
            "The embedding request failed. Confirm that Ollama "
            f"is running and that '{EMBEDDING_MODEL}' is installed. "
            f"Original error: {exc}"
        ) from exc


def check_ollama_connection() -> str:
    """Check whether the local Ollama server is reachable."""

    try:
        ollama.list()
        return "Ollama is running and reachable."

    except Exception as exc:
        raise RuntimeError(
            "Ollama could not be reached. Start the Ollama "
            "application or run 'ollama serve'. "
            f"Original error: {exc}"
        ) from exc