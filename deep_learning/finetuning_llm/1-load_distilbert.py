#!/usr/bin/env python3
"""This module loads a tokenizer and a sequence classification
model for text classification.
"""
import transformers


def load_distilbert(model_name, num_classes, id2label, label2id):
    """
    Load a tokenizer and a sequence classification model
    for text classification.

    Args:
        model_name: Name of the pre-trained DistilBERT to load.
        num_classes: Total number of output classes.
        id2label: Dictionary mapping numeric label IDs
                  to human-readable label names.
        label2id: Dictionary mapping label names
                  to their corresponding numeric IDs.

    Returns:
        The tokenizer for preprocessing text inputs.
        The sequence classification model.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
        )

    return (tokenizer, model)
