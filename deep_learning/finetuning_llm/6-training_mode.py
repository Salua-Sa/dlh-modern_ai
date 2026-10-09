#!/usr/bin/env python3
"""This module trains, evaluates, and saves a Hugging Face model.
"""
import transformers


def train_model(model, training_args, train_dataset, eval_dataset,
                test_dataset, tokenizer, data_collator,
                compute_metrics, model_save_name):
    """
     Train, evaluate, and save a Hugging Face model.

    Args:
        model: Pre-trained model to fine-tune.
        training_args: Training arguments with hyperparameters
                       and strategies.
        train_dataset: Tokenized training dataset.
        eval_dataset: Tokenized validation dataset.
        test_dataset: Tokenized test dataset.
        tokenizer: Tokenizer (or processing class) used
                   for preprocessing.
        data_collator: Data collator for dynamic padding.
        compute_metrics: Function that computes evaluation metrics
                         from predictions and labels.
        model_save_name: Directory to save the trained model and tokenizer.

    Returns:
        trainer: Hugging Face Trainer instance.
        train_results: Training metrics and logs.
        test_results: Evaluation results on the test dataset.
    """
    trainer = transformers.Trainer(model=model,
                                   args=training_args,
                                   train_dataset=train_dataset,
                                   eval_dataset=eval_dataset,
                                   data_collator=data_collator,
                                   compute_metrics=compute_metrics)

    train_results = trainer.train()
    test_results = trainer.evaluate(test_dataset)

    trainer.save_model(model_save_name)
    tokenizer.save_pretrained(model_save_name)

    trainer.push_to_hub()

    return (trainer, train_results, test_results)
