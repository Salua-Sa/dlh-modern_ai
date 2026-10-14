#!/usr/bin/env python3
"""This module loads a HuggingFace embedding model using LangChain.
"""
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """
    Load a HuggingFace embedding model using LangChain,
    which can be used to generate vector representations
    of documents for tasks like similarity search or retrieval.

    Args:
        model_name: A string representing the name of the HuggingFace
                    model to use for generating embeddings.

    Returns:
        An instance of HuggingFaceEmbeddings to compute embeddings.
    """
    embedding_documents = HuggingFaceEmbeddings(model_name=model_name)

    return embedding_documents
