#!/usr/bin/env python3
"""
This module applies YOLO-compatible Albumentations-exclusive
data augmentation.
"""
import albumentations
import numpy as np


def custom_aug(image, bboxes, labels):
    """
    Apply YOLO-compatible data augmentation using Albumentations.
    The function must apply the following transformations:
    - motion blur (blur_limit=5, p=0.9)
    - With p= 0.9, One of these will be applied:
         - elastic (alpha=1, sigma=50, p=0.2)
         - or optical distortions (distort_limit=0.05, p=0.2)

    Args:
        image (np.ndarray): Input image
        bboxes (List[List[int]]): Bounding boxes in Pascal VOC format
        labels (List[int]): Class labels corresponding to each bounding box

    Returns:
        the augmented image np.ndarray,
        augmented bounding boxes np.ndarray and labels List[int]
    """

    # Create the custom augmentation pipeline
    transform = albumentations.Compose(
        [
            # Apply motion blur with 90% probability
            albumentations.MotionBlur(
                blur_limit=5,
                p=0.9
            ),

            # Apply one of the following distortions
            albumentations.OneOf(
                [
                    # Elastic deformation
                    albumentations.ElasticTransform(
                        alpha=1,
                        sigma=50,
                        p=0.2
                    ),

                    # Optical distortion
                    albumentations.OpticalDistortion(
                        distort_limit=0.05,
                        p=0.2
                    )
                ],
                p=0.9
            )
        ],

        # Bounding boxes are in Pascal VOC format
        bbox_params=albumentations.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        ),

        # Make augmentation reproducible
        seed=42
    )

    # Apply augmentation
    augmented = transform(
        image=image,
        bboxes=bboxes,
        labels=labels
    )

    # Get the augmented results
    augmented_image = np.array(augmented["image"])
    augmented_bboxes = np.array(augmented["bboxes"])
    augmented_labels = augmented["labels"]

    return augmented_image, augmented_bboxes, augmented_labels
