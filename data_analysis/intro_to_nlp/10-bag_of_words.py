#!/usr/bin/env python3
"""
This module builds a Bag-of-Words feature matrix from a list of token lists.
"""
import sklearn


def bag_of_words(corpus_tokens, max_features=5000, ngram_range=(1, 2),
                 min_df=2, max_df=0.95, binary=False):
    """
    Build a bag of words feature matrix.

    Arguments:
        corpus_tokens: List of token lists
        max_features: Maximum number of features
        ngram_range: Range of n-grams to include
        min_df: minimum document frequency
        max_df: maximum document frequency
        binary: use 0/1 when it is True

    Returns:
        X: the sparse feature matrix (n_samples, n_features).
        vectorizer: the fitted CountVectorizer object.
    """
    # Join each token list into a whitespace-separated string.
    document = []
    for tokens in corpus_tokens:
        document.append(" ".join(tokens))
    # Create the CountVectorizer
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        binary=binary
        )

    # Learn the vocavolary and transform messages into a numerical bow matrix
    X = vectorizer.fit_transform(document)

    return (X, vectorizer)
