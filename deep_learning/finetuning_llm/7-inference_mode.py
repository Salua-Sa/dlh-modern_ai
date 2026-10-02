#!/usr/bin/env python3
"""This module initializes a text classification pipeline
using a fine-tuned Hugging Face model.
"""
import transformers


def inference_mode(model_path, top_k):
    """
    Initialize a text classification pipeline
    using a fine-tuned Hugging Face model.

    Args:
        model_path (str): Path to the saved model and tokenizer.
        top_k (int): Number of top predictions to return per input.

    Returns:
        A transformers.Pipeline object ready to classify new texts.
    """
    classifier = transformers.pipeline("text-classification",
                                       model=model_path,
                                       tokenizer=model_path,
                                       top_k=top_k)

    return classifier
