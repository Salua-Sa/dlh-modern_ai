#!/usr/bin/env python3
"""
This module trains Word2Vec and returns per-message embeddings.
"""
import numpy as np
import gensim.models


def word2vec_embeddings(corpus_tokens, vector_size=50, window=5,
                        min_count=2, sg=0, epochs=10, workers=4):
    """
    Train a Word2Vec model and create one embedding per message

    Arguments:
        corpus_tokens: List of token lists.
        vector_size: Size of each word vector.
        window: Context window size.
        min_count: Minimum number of occurences for a word
        sg: 0 for CBOW, 1 for Skip-gram
        epochs: Number of training epochs
        workers: Number of worker threads

    Returns:
        X is a np.ndarray of shape (n_messages, vector_size).
        model is the trained Word2Vec model.
    """
    # Train gensim Word2Vec on corpus_tokens.
    model = gensim.models.Word2Vec(
        sentences=corpus_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
        )

    # Represent each message as the mean of its in-vocab token vectors.
    message_vectors = []
    for tokens in corpus_tokens:
        word_vectors = []
        for token in tokens:
            # Tokens not in the Word2Vec vocabulary are silently ignored.
            if token in model.wv:
                word_vectors.append(model.wv[token])
            # If a message has no in-vocab tokens, its row is a zero vector.
        if word_vectors:
            message_vector = np.mean(word_vectors, axis=0)
        else:
            message_vector = np.zeros(vector_size)
        message_vectors.append(message_vector)

    X = np.array(message_vectors)

    return (X, model)
