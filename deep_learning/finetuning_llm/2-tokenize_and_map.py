#!/usr/bin/env python3
"""This module tokenizes the text inputs of the entire
Emotion dataset using a pretrained tokenizer.
"""


def tokenize_and_map(dataset, tokenizer, max_length, truncation, batched):
    """
    Tokenize the text inputs of the entire Emotion dataset
    using a pretrained tokenizer.

    Args:
        dataset: The dataset with 'train', 'validation', and 'test' splits.
        tokenizer: Pretrained tokenizer.
        max_length: Maximum token length for truncation.
        truncation: Whether to truncate sequences longer than max_length.
        batched: Whether to process the dataset in batches.

    Returns:
        The tokenized train, validation, and test splits
    """
    # Tokenize the text
    def tokenize(text):
        tokenized_text = tokenizer(text["text"],
                                   truncation=truncation,
                                   max_length=max_length
                                   )
        return tokenized_text
    # Apply the tokanization to the whole dataset
    tokenized_dataset = dataset.map(tokenize,
                                    batched=batched)

    # Get dataset split
    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]
    test_dataset = tokenized_dataset["test"]

    return (train_dataset, validation_dataset, test_dataset)
