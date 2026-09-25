#!/usr/bin/env python3
""" This module creates a high-level interface to perform image
classification using a pre-trained large language model adapted
for computer-vision applications.
"""
import transformers


def image_classifier(model):
    """
    Creates a high-level interface to perform image classification
    using a pre-trained large language model adapted for computer-vision
    applications.

    Args:
        model (str): Name of the pre-trained to use.

    Returns:
        classifier: A Hugging Face pipeline object.
    """
    classifier = transformers.pipeline("image-classification",
                                       model=model)

    return classifier
