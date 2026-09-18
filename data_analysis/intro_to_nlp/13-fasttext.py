#!/usr/bin/env python3
"""
This module trains FastText and returns per-message embeddings.
"""
import numpy as np
import gensim.models


def fasttext_embeddings(corpus_tokens, vector_size=50, window=5,
                        min_count=1, sg=0, epochs=10, workers=4):
    """
    Train a FastText model and create one embedding per message

    Arguments:
        corpus_tokens: List of token lists.
        vector_size: Size of each word vector.
        window: Context window size.
        min_count: Minimum number of occurences for a word
        sg: 0 for CBOW, 1 for Skip-gram
        epochs: Number of training epochs
        workers: Number of worker threads

    Returns:
        X is a np.ndarray of shape (n_messages, vector_size)
        model is the trained FastText model.
    """
    # Train a FastText model on corpus_tokens
    model = gensim.models.FastText(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
        )
    # Represent each message as the mean of its token vectors.
    message_vectors = []
    for tokens in corpus_tokens:
        word_vectors = []
        for token in tokens:
            # FastText uses subword n-grams,
            # so every token has a vector including OOV tokens
            try:
                word_vectors.append(model.wv[token])
            except KeyError:
                pass
            # If a message has no in-vocab tokens, its row is a zero vector.
        if word_vectors:
            message_vector = np.mean(word_vectors, axis=0)
        else:
            message_vector = np.zeros(vector_size)
        message_vectors.append(message_vector)

    X = np.array(message_vectors)

    return (X, model)
