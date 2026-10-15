#!/usr/bin/env python3
"""This module combines multiple documents into a coherent context and
generates answers from them, while leveraging prior conversation history.
"""
from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(llm, system_message, human_message):
    """
    Combine multiple documents into a coherent context and generates
    answers from them, while leveraging prior conversation history.

    Args:
        llm: An instance of a language model to generate answers.
        system_message: String defining instructions for the assistant.
        human_message: String containing the user’s input or question

    Returns:
        A ChatPromptTemplate used for generating answers.
        A LangChain document chain that sends documents,
                    chat history, and user input to the LLM.
    """
    chat_prompt = prompts.ChatPromptTemplate.from_messages(
        [("system", system_message),
         prompts.MessagesPlaceholder(variable_name="chat_history"),
         ("human", human_message),
    ])

    langChain_document = combine_documents.create_stuff_documents_chain(
        llm,
        prompt=chat_prompt)

    return (chat_prompt, langChain_document)
