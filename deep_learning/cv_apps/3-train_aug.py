#!/usr/bin/env python3
"""
This module trains a YOLO model using data augmentation, and allows
custom Albumentations transforms to be applied during training.
"""
from ultralytics import YOLO


def train_with_augmentation(data, model_path="yolov8n.pt", epochs=100,
                            imgsz=640, batch=16, augmentation=True,
                            yolo_aug_params=None,
                            albumentations_transforms=None,
                            save=False, plots=False, verbose=False):
    """
    Train a YOLO model using data augmentation, and allows custom
    Albumentations transforms to be applied during training.

    Args:
        data (str): Path to the dataset YAML file containing
            train/val/test paths and class names for Ultralytics YOLO.
        model_path Path to pre-trained weights file (e.g., 'yolov8n.pt')
            or model configuration file.
        epochs: Number of training epochs.
        imgsz (int or tuple, optional): Input image size for training.
            Can be a single integer (square) or tuple (height, width).
        batch: Batch size for training.
        augmentation: Global flag to enable/disable augmentation.
        yolo_aug_params: Dictionary of YOLO's native augmentation
            parameters to customize built-in transforms. When provided,
            these parameters override the default augmentation settings.
        albumentations_transforms: List of Albumentations transform objects
            for custom augmentation pipeline. When provided, this takes
            precedence over YOLO's built-in augmentations.
        save: Whether to save training checkpoints and final model.
        plots: Whether to generate and save training plots.
        verbose: Whether to display detailed training progress and output.

    Returns:
        the trained yolo model and the full training output.
    """

    # Load the YOLO model
    model = YOLO(model_path)

    # Store the basic training parameters
    train_params = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "augment": augmentation,
        "save": save,
        "plots": plots,
        "verbose": verbose
    }

    # Add custom YOLO augmentation parameters if provided
    if yolo_aug_params is not None:
        train_params.update(yolo_aug_params)

    # Train the YOLO model
    results = model.train(**train_params)

    return model, results
