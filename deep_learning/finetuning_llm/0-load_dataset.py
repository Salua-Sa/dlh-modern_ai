#!/usr/bin/env python3
"""This module loads the Emotion dataset
using the Hugging Face datasets library.
"""
import datasets


def load_emotion_dataset():
    """
    Load the Emotion dataset using the Hugging Face datasets library.

    Returns:
        The full dataset object containing the train,
        validation, and test splits.
    """
    dataset = datasets.load_dataset("dair-ai/emotion")

    return dataset
