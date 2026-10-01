#!/usr/bin/env python3
"""This module calculates key evaluation metrics
for a classification model.
"""
import numpy as np
import sklearn.metrics


def compute_metrics(predictions):
    """
    Calculate key evaluation metrics for a classification model.

    Args:
        predictions: A transformers.EvalPrediction object.

    Returns:
        Dictionary containing the computed metrics:
        {
        accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float
        }
    """
    logits = predictions.predictions
    labels = predictions.label_ids
    predicted_labels = np.argmax(logits, axis=1)

    accuracy = sklearn.metrics.accuracy_score(labels,
                              predicted_labels
                              )

    precision = sklearn.metrics.precision_score(labels,
                                predicted_labels,
                                average="weighted"
                                )

    recall = sklearn.metrics.recall_score(labels,
                          predicted_labels,
                          average="weighted"
                          )

    f1 = sklearn.metrics.f1_score(labels,
                  predicted_labels,
                  average="weighted"
                  )

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }
