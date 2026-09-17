#!/usr/bin/env python3
"""
This module generates n-grams from a token list.
"""
import nltk


def generate_ngrams(tokens, n=2):
    """
    Generate n-grams from a list of tokens.

    Arguments:
        tokens (list[str]): List of tokens used to generate n-grams.
        n (int): Size of each n-gram.

    Returns:
        A list of strings where each n-gram is n consecutive tokens
        joined with "_".
    """
    # Return [] if tokens is not a list or has fewer than n elements.
    if not isinstance(tokens, list) or len(tokens) < n:
        return []

    # Use nltk to generate n-grams.
    ngrams = nltk.ngrams(tokens, n)

    ngrams_list = []

    for gram in ngrams:
        joined_gram = "_".join(gram)

        ngrams_list.append(joined_gram)
