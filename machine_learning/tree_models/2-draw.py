#!/usr/bin/env python3
"""
This module displays the textual structure of a trained
decision tree classifier using Scikit-learn.
"""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Displays the textual structure of a trained decision
    tree classifier using Scikit-learn.

    Args:
        clf: A trained DecisionTreeClassifier instance from Scikit-learn.
        feature_names: A list of the input feature names.
        class_names: A list of the target class names.

    Returns:
        None.
    """
    text_clf = tree.export_text(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names)
        )
    print(text_clf)
