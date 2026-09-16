#!/usr/bin/env python3
"""
This module removes stopwords from a token list.
"""
import nltk


def remove_stopwords(tokens, language="english",
                     extra_words=None, keep_words=None):
    """
    Remove stopwords from a list of tokens

    Arguments:
        tokens: List of tokens to filter.
        language: NLTK stopword language to load. Defaults to "english".
        extra_words: Additional words to add to the stopword set.
        keep_words: Words to exclude from the stopword set

    Returns:
        The filtered token list.
    """
    # Return [] if tokens is not a list.
    if not isinstance(tokens, list):
        return []

    # Load the NLTK stopword list for the given language
    stop_words = set(nltk.corpus.stopwords.words(language))

    # Add any words in extra_words to the stopword set
    if extra_words is not None:
        stop_words.update(extra_words)

    # Remove any words in keep_words from the stopword set
    if keep_words is not None:
        stop_words.difference_update(keep_words)

    final_tokens = []
    for token in tokens:
        if token.lower() not in stop_words:
            final_tokens.append(token)

    return final_tokens
