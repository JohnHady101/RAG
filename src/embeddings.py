"""Text embeddings via the Gemini API.

The client is created lazily so importing this module never needs
network access or an API key.
"""

from google import genai

from config import EMBEDDING_MODEL, GEMINI_API_KEY

_client = None


def get_client():
    """Return a shared genai client, creating it on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_embedding(text: str) -> list[float]:
    """Embed a single text and return its vector (length EMBEDDING_DIM)."""
    result = get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values
