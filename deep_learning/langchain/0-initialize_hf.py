#!/usr/bin/env python3
"""This module sets up a Hugging Face text-to-text
generation language model using LangChain.
"""
import transformers
import langchain-huggingface


def initialize_hf_llm(model_name, max_tokens):
    """
    Set up a Hugging Face text-to-text generation
    language model using LangChain.

    Args:
        model_name: Name of the Hugging Face model to load.
        max_tokens: Maximum number of tokens to generate for the output.

    Returns:
        llm: An instance of HuggingFacePipeline.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipeline_hf = langchain-huggingface.pipeline("text2text-generation",
                                                 model=model,
                                                 tokenizer=tokenizer,
                                                 max_new_tokens=max_tokens)

    llm = transformers.HuggingFacePipeline(pipeline=pipeline_hf)

    return llm
