#!/usr/bin/env python3
"""This module sets up a reusable chat-style prompt.
"""
from langchain import prompts


def create_chat_prompt_template(system_message, human_message):
    """
    Set up a reusable chat-style prompt.

    Args:
        system_message: String defining the assistant’s instructions.
        human_message: String containing the user’s input with placeholders.

    Returns:
        ChatPromptTemplate: An object that can be formatted dynamically
                            with variables.
    """
    chat_prompt_template = prompts.ChatPromptTemplate(
        [("system", system_message),
         ("human", human_message),])

    return chat_prompt_template
