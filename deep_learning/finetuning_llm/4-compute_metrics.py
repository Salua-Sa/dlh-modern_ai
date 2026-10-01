#!/usr/bin/env python3
"""This module calculates key evaluation metrics
for a classification model.
"""
import numpy as np
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score
                             )


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

    accuracy = accuracy_score(labels,
                              predicted_labels
                              )

    precision = precision_score(labels,
                                predicted_labels,
                                average="weighted"
                                )

    recall = recall_score(labels,
                          predicted_labels,
                          average="weighted"
                          )

    f1 = f1_score(labels,
                  predicted_labels,
                  average="weighted"
                  )

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
            }
