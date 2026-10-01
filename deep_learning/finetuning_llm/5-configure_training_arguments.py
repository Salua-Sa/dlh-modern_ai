#!/usr/bin/env python3
"""This module sets up key hyperparameters/arguments
for training a Hugging Face model.
"""
import transformers


def configure_training_args(output_dir, epochs,
                            per_device_train_batch_size,
                            per_device_eval_batch_size,
                            learning_rate, weight_decay,
                            metric_for_best_model, seed):
    """
    Set up key hyperparameters/arguments for training a Hugging Face model.

    Args:
        output_dir: Directory to save checkpoints and logs.
        epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for regularization.
        metric_for_best_model: Metric used to determine the best checkpoint.
        seed: Random seed for reproducibility.

    Returns:
        A transformers.TrainingArguments object ready
        to be used for model training.
    """

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_only_model="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        push_to_hub=True,
        seed=seed
        )

    return training_args
