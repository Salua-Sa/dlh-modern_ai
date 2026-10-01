#!/usr/bin/env python3
"""This module sets up a dynamic padding mechanism
for tokenized inputs using a pretrained tokenizer.
"""
import transformers


def create_data_collator(tokenizer):
    """
    Set up a dynamic padding mechanism for
    tokenized inputs using a pretrained tokenizer.

    Args:
        tokenizer: Pretrained tokenizer.

    Returns:
        DataCollatorWithPadding instance
        that dynamically pads inputs per batch.
    """

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer
        )

    return data_collator
