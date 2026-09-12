#!/usr/bin/env python3
"""
This module performs inference parameter tuning to find optimal confidence
and IoU thresholds for best performance on the validation set.
"""
from ultralytics import YOLO


def tune_inference(data_yaml, model,
                   conf_list=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                   iou_list=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                   imgsz=640):
    """
    Perform inference parameter tuning to find optimal confidence
    and IoU thresholds for best performance on the validation set.

    The function must:
    - Perform grid search over all combinations of conf_thresholds
      and iou_thresholds
    - For each combination, run validation using model.val()
      and collect metrics:
        - mAP50 (mean Average Precision at IoU=0.50)
        - mAP50-95 (mean Average Precision at IoU=0.50:0.95)
        - Precision
        - Recall
        - F1-score Returns a dictionary containing:
            - best_conf: Optimal confidence threshold
            - best_iou: Optimal IoU threshold
            - best_metrics: Dictionary of metrics at optimal settings.
            - all_results: DataFrame or list of dictionaries with all
              tested combinations and their metrics.

    Args:
        model: Path to trained model weights or YOLO model object
        val_images_path: Path to validation images directory
        conf_thresholds: List of confidence thresholds to test
        iou_thresholds: List of IoU thresholds for NMS to test
        imgsz: Image size for inference (default: 640)

    Returns:
        None
    """
    # Load the model if a model path was provided
    if isinstance(model, str):
        model = YOLO(model)

    # Store all tested combinations
    all_results = []

    # Test every confidence threshold
    for conf in conf_list:
        # Test every IoU threshold
        for iou in iou_list:
            # Validate the model with this combination
            metrics = model.val(
                data=data_yaml,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False
            )
            # Get validation metrics
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            precision = metrics.box.mp
            recall = metrics.box.mr
            # Calculate F1-score
            if precision + recall > 0:
                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                )
            else:
                f1 = 0.0

            # Save this combination and its results
            result = {
                "conf": conf,
                "iou": iou,
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            all_results.append(result)

    return all_results
