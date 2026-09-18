#!/usr/bin/env python3
"""
This module generates a word cloud from a preprocessed corpus.
"""
import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """
    Generate a word cloud from a list of token lists.

    Arguments:
        corpus_tokens: Corpus represented as a list of token lists.
        n: Number of top frequent tokens to display.

    Returns:
        The fitted wordcloud object.
    """
    tokens_lists = []

    # Concatenate all tokens into a single string.
    for token_list in corpus_tokens:
        for token in token_list:
            tokens_lists.append(token)
    text = " ".join(tokens_lists)

    # Create a WordCloud with these exact parameters:
    wc = wordcloud.WordCloud(max_words=max_words,
                             background_color="white",
                             width=800,
                             height=400,
                             random_state=42)
    wc.generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if label:
        plt.title(f"WordCloud – {label}")
    else:
        plt.title("WordCloud")
    plt.tight_layout()
    plt.show()

    return wc
