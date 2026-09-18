#!/usr/bin/env python3
"""
This module plots the most frequent tokens in a preprocessed corpus.
"""
import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """
    Plot the n most frequent tokens in a corpus.

    Arguments:
        corpus_tokens: Corpus represented as a list of token lists.
        n: Number of top frequent tokens to display.

    Returns:
        The full frequency distribution object.
    """
    tokens_lists = []

    # Flatten the list of lists into a single token list & computes frequencies
    for token_list in corpus_tokens:
        for token in token_list:
            tokens_lists.append(token)

    freq_dist = nltk.FreqDist(tokens_lists)
    most_common = freq_dist.most_common(n)

    words = []
    frequencies = []

    for word, frequency in most_common:
        words.append(word)
        frequencies.append(frequency)

    # Plot a bar chart using plt.bar
    plt.figure(figsize=(12, 5))
    plt.bar(words, frequencies)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    return freq_dist
