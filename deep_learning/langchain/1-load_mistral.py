#!/usr/bin/env python3
"""This module initialize a Mistral language model using LangChain.
"""
import langchain_mistralai


def load_mistral(model_name, temperature):
    """
    Initialize a Mistral language model using LangChain.

    Args:
        model_name: the name of the Mistral model to use.
        temperature: A float controlling the randomness of the model’s output.

    Returns:
        llm: An instance of ChatMistralAI.
    """
    llm = langchain_mistralai.ChatMistralAI(
        model=model_name,
        temperature=temperature,
    )

    return llm
