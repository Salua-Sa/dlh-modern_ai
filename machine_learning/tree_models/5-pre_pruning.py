#!/usr/bin/env python3
"""
This module uses Scikit-learn to perform a Grid Search for the best
pre-pruning hyperparameters for a decision tree classifier.
"""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Uses Scikit-learn to perform a Grid Search for the best
    pre-pruning hyperparameters for a decision tree classifier.

    The search explores the following hyperparameters:
        - criterion: "gini" or "entropy".
        - max_depth: integer values in the range (2, 5).
        - min_samples_leaf: integer values in the range (2, 5).
        - min_samples_split: integer values in the range (2, 5).

    Args:
        X: Input features.
        y: Target labels.
        clf: An untrained DecisionTreeClassifier instance.

    Returns:
        A dictionary containing the best combination
        of hyperparameters found during the grid search.
    """
    parameters = {
        "criterion": ["gini", "entropy"],
        "max_depth": list(range(2, 5)),
        "min_samples_leaf": list(range(2, 5)),
        "min_samples_split": list(range(2, 5))
        }

    search = model_selection.GridSearchCV(
        clf,
        parameters)

    search.fit(X, y)
    return search.best_params_
