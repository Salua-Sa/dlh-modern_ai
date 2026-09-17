#!/usr/bin/env python3
"""
This module normalises tokens via lemmatisation or stemming.
"""
import re
import nltk


PLACEHOLDER_RE = re.compile(r'^<[A-Za-z]+>$')


def normalize_tokens(tokens, method="lemmatize"):
    """
    Normalize tokens using lemmatization or stemming.

    Arguments:
        tokens (list[str]): List of tokens to normalize.
        method (str): Normalization method. Must be "lemmatize" or "stem".

    Returns:
        a list of normalized tokens.
    """
    normalize_tokens = []

    # Raise ValueError("method must be 'lemmatize' or 'stem'") for any other value.
    if method not in ["stem", "lemmatize"]:
        raise ValueError("method must be 'lemmatize' or 'stem'")

    # For method="stem": apply PorterStemmer` to each token, skipping placeholders.
    if method == "stem":
        stemmer = nltk.stem.PorterStemmer()
        for token in tokens:
            if PLACEHOLDER_RE.fullmatch(token):
                normalize_tokens.append(token)
                continue
            stemmed_token = stemmer.stem(token)
            normalize_tokens.append(stemmed_token)

        return normalize_tokens

    # For method="lemmatize": apply POS-aware lemmatisation using WordNetLemmatizer`.
    if method == "lemmatize":
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tagged_tokens = nltk.pos_tag(tokens)
        for token, tag in tagged_tokens:
            if PLACEHOLDER_RE.fullmatch(token):
                normalize_tokens.append(token)
                continue
            pos = get_pos(tag)
            lemmatize = lemmatizer.lemmatize(token, pos)
            normalize_tokens.append(lemmatize)

        return normalize_tokens


def get_pos(tag):
    """
    Convert an NLTK POS tag to a WordNet POS tag
    """
    if tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith("N"):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN
