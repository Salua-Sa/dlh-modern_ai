#!/usr/bin/env python3
"""This module generates a search query based on the conversation
history and the user’s current input to retrieve the most relevant documents.
"""
from langchain_core import prompts
from langchain import chains


def history_aware_retriever(llm, retriever, human_message):
    """
    Generate a search query based on the conversation history
    and the user’s current input to retrieve the most relevant documents.

    Args:
        llm: An instance of a language model to help generate the search query.
        retriever: LangChain retriever object that performs similarity search.
        human_message: the user’s input or question .

    Returns:
        A ChatPromptTemplate object for generating search queries.
        A LangChain history-aware retriever.
    """
    chat_prompt = prompts.ChatPromptTemplate.from_messages([
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message),])

    history_retriever = (chains.history_aware_retriever.
                         create_history_aware_retriever(llm,
                                                        retriever,
                                                        chat_prompt))

    return (chat_prompt, history_retriever)
