#!/usr/bin/env python3
"""This module wraps a conversational retrieval chain
to support session-aware chat histories.
"""
from langchain_core import runnables


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """
    Wrap a conversational retrieval chain to support
    session-aware chat histories.

    Args:
        retrieval_chain: A conversational retrieval chain.
        get_session_history: A function that returns a chat history object.
        input_key: A string representing the user input.
        history_key: A string where the chat history is stored.
        output_key: A string where the model output should be stored.

    Returns:
        A RunnableWithMessageHistory.
    """
    chain_session = runnables.RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
        output_messages_key=output_key)

    return chain_session
