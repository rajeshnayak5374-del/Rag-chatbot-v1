# rag_pipeline/embeddings.py
# Author: Rajesh (Junior AI Engineer)
# Ticket: SAGE-001
# Purpose: Embedding model wrapper
#          Loads HuggingFace MiniLM once and reuses
#          across ingest and retrieve pipelines

from langchain_huggingface import HuggingFaceEmbeddings

# Model selection
# Development: all-MiniLM-L6-v2 — free, local, no API key needed
# Production upgrade: text-embedding-3-large via Azure OpenAI
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Singleton instance — loaded once, reused everywhere
# Loading this model takes 20-30 seconds and downloads 90MB
# Without singleton: every function call reloads model — very slow
# With singleton: load once on first call, instant on all subsequent calls
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model instance.
    Uses singleton pattern — loads only on first call.

    Returns:
        HuggingFaceEmbeddings: loaded embedding model
    """
    global _embeddings
    if _embeddings is None:
        print(f"[embeddings] Loading model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        print(f"[embeddings] Model loaded. Dimensions: {EMBEDDING_DIMENSIONS}")
    return _embeddings


def embed_text(text: str) -> list:
    """
    Convert single text string to embedding vector.
    Used by retrieve.py for query embedding.

    Args:
        text: input text string

    Returns:
        list: embedding vector of 384 floats
    """
    return get_embeddings().embed_query(text)


def embed_documents(texts: list) -> list:
    """
    Convert list of texts to embedding vectors in batch.
    Used by ingest.py for chunk embedding.
    Batch processing is faster than calling one by one.

    Args:
        texts: list of text strings

    Returns:
        list: list of embedding vectors
    """
    return get_embeddings().embed_documents(texts)