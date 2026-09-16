#!/usr/bin/env python3
"""
This module removes low-information tokens.
"""
import re


PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')


def filter_tokens(tokens, min_len=2, strip_hashtag=False):
    """
    Remove low-information tokens from a list of tokens.

    Arguments:
        tokens (list[str]): List of tokens to filter.
        min_len (int): Minimum token length to keep.
        strip_hashtag: If True, removes leading # from hashtags
                       before processing.

    Returns:
        the filtered token list
    """
    filtered_tokens = []

    # Return [] for an empty or falsy tokens input.
    if not tokens:
        return []
    # Keep any token that matches PLACEHOLDER_RE.
    for token in tokens:
        if PLACEHOLDER_RE.fullmatch(token):
            filtered_tokens.append(token)
            continue

        # If strip_hashtag=True and the token starts with #,
        # strip the # prefix before continuing
        if strip_hashtag and token.starstwith("#"):
            token = token[1:]

        # Drop tokens shorter than min_len.
        if len(token) < min_len:
            continue

        # Drop tokens that contain no alphabetic characters
        has_letter = False
        for character in token:
            if character.isalpha():
                has_letter = True
                break
        if has_letter is False:
            continue

        # Keep token passed all filters
        filtered_tokens.append(token)

    return filtered_tokens
