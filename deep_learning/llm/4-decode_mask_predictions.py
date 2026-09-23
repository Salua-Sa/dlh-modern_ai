#!/usr/bin/env python3
""" This module converts the logits for all <mask> tokens
into their corresponding vocabulary tokens.
"""


def decode_mask_predictions(mask_logits_list, tokenizer):
    """
    Convert the logits for all <mask> tokens into their
    corresponding vocabulary tokens.

    Args:
        mask_logits_list: A list of logits tensors.
        tokenizer: A RoBERTa tokenizer instance.

    Returns:
        decoded_tokens: A list of lists, where each inner list contains all
        vocabulary tokens corresponding to the logits at each mask position.
    """
    decoded_tokens = []

    for mask_logits in mask_logits_list:
        mask_logits_size = len(mask_logits)
        token_ids = range(mask_logits_size)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        decoded_tokens.append(tokens)

    return decoded_tokens
