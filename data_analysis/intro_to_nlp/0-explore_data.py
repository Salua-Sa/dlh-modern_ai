#!/usr/bin/env python3
"""
This module performs initial dataset exploration.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def explore_data(df):
    """
    Performs initial dataset exploration:
    Creates a figure with two subplots side by side: 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    Left subplot: bar chart of ham vs spam counts using sns.barplot:
        - title: "Ham vs Spam Counts", xlabel: "label", ylabel: "count"
    Right subplot: histogram of raw message lengths using sns.histplot:
        - bins: 50
        - title: "Histogram of Raw Message Lengths", xlabel: "length", ylabel: "count"

    Args:
        df: Dataset containing 'label' and 'message' columns.
    Returns:
        None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Configuration left subplot
    label_count = df['label'].value_counts()

    sns.barplot(x=label_count.index,
                y=label_count.values,
                ax=ax1)
    ax1.set_title("Ham vs Spam Counts")
    ax1.set_xlabel("label")
    ax1.set_ylabel("count")

    # Configuration right subplot
    message_lengths = df['message'].str.len()
    sns.histplot(message_lengths,
                 bins=50,
                 ax=ax2)
    ax2.set_title("Histogram of Raw Message Lengths")
    ax1.set_xlabel("length")
    ax1.set_ylabel("count")

    # Prevent labels and plots from overlapping
    plt.tight_layout()

    # Display the figure
    plt.show()

    return None
