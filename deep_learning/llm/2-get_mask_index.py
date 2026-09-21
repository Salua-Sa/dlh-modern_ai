#!/usr/bin/env python3
""" This module identifies and returns the positions
of all <mask> tokens in the tokenized input sequence.
"""


def get_mask_index(inputs, tokenizer):
    """
    Identifies and returns the positions of all <mask> tokens
    in the tokenized input sequence.

    Args:
        inputs: Tokenized inputs.
        tokenizer: An instance of RobertaTokenizer.

    Returns:
        mask_indices: A list containing the index of every <mask>
                      token found in the sequence.
    """
    token_ids = inputs["input_ids"][0]
    mask_token_id = tokenizer.mask_token_id
    mask_indices = []

    # The function searches through the input token IDs and extracts the indices where mask tokens appear
    for index, token_id in enumerate(token_ids):
        if token_id.item() == mask_token_id:
            mask_indices.append(index)
        # If no mask token is found, the function must raise a ValueError
        if not mask_indices:
            raise ValueError("No <mask> token found in the input!")

    return mask_indices
