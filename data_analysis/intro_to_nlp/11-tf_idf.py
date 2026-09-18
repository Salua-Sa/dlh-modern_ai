#!/usr/bin/env python3
"""
This module builds a TF-IDF feature matrix from a list of token lists.
"""
import sklearn


def tf_idf(corpus_tokens, max_features=5000, ngram_range=(1, 2),
           min_df=2, max_df=0.95, norm='l2'):
    """
    The function should:

    Arguments:
        corpus_tokens
        max_features=5000
        ngram_range=(1, 2)
        min_df=2
        max_df=0.95
        norm='l2'

    Returns:
         X is the sparse TF-IDF feature matrix (n_samples, n_features).
         vectorizer is the fitted TfidfVectorizer object.
    """
    #  and the remaining parameters passed through.
    # Join each token list into a whitespace-separated string.
    document = []
    for tokens in corpus_tokens:
        document.append(" ".join(tokens))
    # Create the CountVectorizer
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        norm=norm
        )

    # Learn the vocavolary and transform messages into a numerical bow matrix
    X = vectorizer.fit_transform(document)

    return (X, vectorizer)
