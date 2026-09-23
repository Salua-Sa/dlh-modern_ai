#!/usr/bin/env python3
""" This module creates a high-level interface for performing
Masked Language Modeling using a pre-trained large language model.
"""
import transformers


def fill_mask(model_name, top_k):
    """
    Create a high-level interface for performing Masked Language
    Modeling using a pre-trained large language model.

    Args:
        model_name: Name of the pre-trained model to use.
        top_k: Number of top predictions.

    Returns:
        fill: A Hugging Face pipeline object.
    """
    fill = transformers.pipeline("fill-mask",
                                 model=model_name,
                                 top_k=top_k)

    return fill
