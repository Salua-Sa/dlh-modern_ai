#!/usr/bin/env python3
"""
This module tokenizes a cleaned SMS message.
"""
import nltk


EMOTICON_MAP = {
    "<3":   "<EMO>", "</3": "<EMO>",
    ":)":   "<EMO>", ":-)": "<EMO>",
    ":(":   "<EMO>", ":-(": "<EMO>",
    ":d":   "<EMO>", ";)":  "<EMO>",
    ":|":   "<EMO>", ">:(": "<EMO>",
    ":p":   "<EMO>", "b)":  "<EMO>",
    "o:)":  "<EMO>",
}


def normalize_emoticons(tokens, emoticon_action="replace"):
    if not isinstance(tokens, list):
        return []

    result = []

    for token in tokens:
        mapped = EMOTICON_MAP.get(token.lower())

        if mapped:
            if emoticon_action == "replace":
                result.append(mapped)
        else:
            result.append(token)

    return result


def tokenize_text(text, method="tweet"):
    """
    Tokenize a cleaned SMS message using the selected method
    """
    # Return an empty list if text is not a string
    if not isinstance(text, str):
        return []

    # "tweet": Use NLTK’s TweetTokenizer to limit repeated characters
    if method == 'tweet':
        tokenizer = nltk.TweetTokenizer(reduce_len=True)
        tokens = tokenizer.tokenize(text)
    # "word": Use NLTK’s standard punctuation-aware word tokenization.
    elif method == "word":
        tokens = nltk.word_tokenize(text)
    # "split": Use Python’s basic whitespace splitting.
    elif method == "split":
        tokens = text.split()
    else:
        raise ValueError("Invalid tokenizer method")

    return tokens
