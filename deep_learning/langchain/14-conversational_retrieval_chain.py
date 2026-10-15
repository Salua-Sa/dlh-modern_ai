#!/usr/bin/env python3
"""This module builds a retrieval chain capable of answering user queries.
"""
from langchain import chains


def create_conversational_retrieval_chain(history_aware_retriever,
                                          document_chain):
    """
    Bbuild a retrieval chain capable of answering user queries by fetching
    relevant documents while incorporating prior conversation context.

    Args:
        history_aware_retriever: A retriever-like object that leverages
                  the query and chat history to find relevant documents.
        document_chain: A chain that combines the retrieved documents into
                  a single context for the language model to generate answers.

    Returns:
        A retrieval chain ready to handle user queries with context-aware
        document retrieval and answer generation.
    """
    retrieval_chain = chains.create_retrieval_chain(history_aware_retriever,
                                                    document_chain)

    return retrieval_chain
