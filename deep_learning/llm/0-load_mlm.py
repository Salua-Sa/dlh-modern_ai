#!/usr/bin/env python3
""" This module loads a pre-trained RoBERTa model ready
for Masked Language Modeling (MLM) using Hugging Face Transformers.
"""
from transformers import RobertaForMaskedLM


def load_mlm(model_name):
    """
    Load a pre-trained RoBERTa model ready for Masked Language 
    Modeling (MLM) using Hugging Face Transformers.

    Args:
        model_name (str): Name of the pre-trained model to load.

    Returns:
        model: An instance of RobertaForMaskedLM ready for inference.
    """

    model = RobertaForMaskedLM.from_pretrained(model_name)

    return model
