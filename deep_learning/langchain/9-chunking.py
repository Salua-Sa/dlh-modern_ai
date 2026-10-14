#!/usr/bin/env python3
"""This module splits a list of documents into smaller,
manageable chunks.
"""
from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """
    Split a list of documents into smaller, manageable chunks.
    Use LangChain's RecursiveCharacterTextSplitter to divide the text.

    Args:
        docs: List of LangChain Document objects to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        list: A list of Document objects representing the split chunks
    """
    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )

    chunks = splitter.split_documents(docs)

    return chunks
