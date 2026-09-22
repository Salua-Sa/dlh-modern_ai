#!/usr/bin/env python3
""" This module  computes the raw logits for all <mask> tokens in a
tokenized sentence using a pre-trained RoBERTa masked language model.
"""
import torch


def compute_mask_logits(model, inputs, mask_indices):
    """
    Compute the raw logits for all <mask> tokens in a tokenized sentence
    using a pre-trained RoBERTa masked language model.


    Args:
        model: A RobertaForMaskedLM model loaded with pre-trained weights.
        inputs: Tokenized inputs.
        mask_indices: Positions of all <mask> tokens in the input sequence.

    Returns:
        mask_logits_list: A list containing logits tensors
                          for each <mask> token.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    mask_logits_list = []

    for mask_index in mask_indices:
        mask_logits = logits[0, mask_index]

        mask_logits_list.append(mask_logits)

    return mask_logits_list
