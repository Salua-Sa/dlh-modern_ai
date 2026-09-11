#!/usr/bin/env python3
"""
This module trains a YOLO model using data augmentation, and allows
custom Albumentations transforms to be applied during training.
"""
from ultralytics import YOLO


def train_with_augmentation(data_yaml, model="yolov8n.pt",
                            aug=None, custom_albu=None, epochs=50,
                            imgsz=640, batch=16):
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
    yolo_model = YOLO(model)

    # Train the model
    results = yolo_model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        augment=True
    )

    return results
