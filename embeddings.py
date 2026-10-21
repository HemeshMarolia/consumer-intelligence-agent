"""Embedding function used by ChromaDB.

The default is a small open-source model (all-MiniLM-L6-v2) that runs locally for free.
It downloads about 80 MB the first time it is used.
"""
from chromadb.utils import embedding_functions


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()
