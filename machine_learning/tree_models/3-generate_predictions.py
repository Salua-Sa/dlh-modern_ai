#!/usr/bin/env python3
"""
This module generates predictions from a trained
tree-based classifier using Scikit-learn.
"""


def generate_predictions(clf, X):
    """
    Generate predictions from a trained tree-based
    classifier using Scikit-learn..

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        A NumPy array containing the predicted class
        labels for the input samples.
    """
    predicted_clf = clf.predict(X)
    return predicted_clf
