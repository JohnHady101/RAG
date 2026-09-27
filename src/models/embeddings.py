"""Text embeddings via the local HuggingFace model.

The Gemini client is still created lazily for answer generation
(`get_client`); only the embedding path uses HuggingFace so importing
this module never needs network access or an API key.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google import genai
from src.config import GEMINI_API_KEY
from models.huggingface import vectorize_texts

_client = None

def get_client():
    """Return a shared genai client, creating it on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_embedding(text: str) -> list[float]:
    """Embed a single text with the local HF model and return its vector."""
    embeddings = vectorize_texts([text])
    vec = embeddings[0]
    if hasattr(vec, "detach"):
        vec = vec.detach()
    if hasattr(vec, "cpu"):
        try:
            vec = vec.cpu()
        except Exception:
            pass
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]

if __name__ == "__main__":
    # test embedding a single text
    text = "Hello, world!"
    embedding = get_embedding(text)
    print(f"embedding for '{text}': {embedding[:5]}... (length={len(embedding)})")