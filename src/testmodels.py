from models import embeddings

if __name__ == "__main__":
    # test embedding a single text
    text = "Hello, world!"
    embedding = embeddings.get_embedding(text)
    print(f"embedding for '{text}': {embedding[:5]}... (length={len(embedding)})")