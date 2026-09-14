#!/usr/bin/env python3
"""
This module download the Pascal VOC 2012 dataset and
organize it into the YOLOv8 format.
"""
import shutil
import xml.etree.ElementTree as ET


# Classes we want to keep
classes = {
    "person": 0,
    "car": 1,
    "bicycle": 2
}

# Pascal VOC folders
images_folder = "VOC2012/JPEGImages"
annotations_folder = "VOC2012/Annotations"

train_file = "VOC2012/ImageSets/Main/train.txt"
val_file = "VOC2012/ImageSets/Main/val.txt"

# Output folders
train_images_output = "datasets/detection/images/train"
train_labels_output = "datasets/detection/labels/train"

val_images_output = "datasets/detection/images/val"
val_labels_output = "datasets/detection/labels/val"

# 1. TRAIN

# Read all training image IDs
with open(train_file, "r") as file:
    train_images = file.read().splitlines()

# Go through every training image
for image_id in train_images:

    # Path to the original image
    image_path = images_folder + "/" + image_id + ".jpg"

    # Path to the XML annotation
    xml_path = annotations_folder + "/" + image_id + ".xml"

    # Read the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    # This list will store YOLO labels
    yolo_labels = []

    # Go through every object in the image
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        # Skip classes we do not need
        if class_name not in classes:
            continue

        # Get the Pascal VOC bounding box
        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # Calculate bounding box width and height
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Calculate bounding box center
        x_center = xmin + box_width / 2
        y_center = ymin + box_height / 2

        # Normalize values between 0 and 1
        x_center = x_center / image_width
        y_center = y_center / image_height

        box_width = box_width / image_width
        box_height = box_height / image_height

        # Get class ID
        class_id = classes[class_name]

        # Create one YOLO label line
        yolo_line = (
            str(class_id)
            + " "
            + str(x_center)
            + " "
            + str(y_center)
            + " "
            + str(box_width)
            + " "
            + str(box_height)
        )

        yolo_labels.append(yolo_line)

    # Copy the image to the training folder
    destination_image = (
        train_images_output
        + "/"
        + image_id
        + ".jpg"
    )

    shutil.copy(image_path, destination_image)

    # Create the corresponding YOLO label file
    destination_label = (
        train_labels_output
        + "/"
        + image_id
        + ".txt"
    )

    with open(destination_label, "w") as file:
        for label in yolo_labels:
            file.write(label + "\n")

# 2. VALIDATION

# Read all validation image IDs
with open(val_file, "r") as file:
    val_images = file.read().splitlines()

# Go through every validation image
for image_id in val_images:

    # Path to the original image
    image_path = images_folder + "/" + image_id + ".jpg"

    # Path to the XML annotation
    xml_path = annotations_folder + "/" + image_id + ".xml"

    # Read the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image size
    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    # This list will store YOLO labels
    yolo_labels = []

    # Go through every object in the image
    for obj in root.findall("object"):

        class_name = obj.find("name").text

        # Skip classes we do not need
        if class_name not in classes:
            continue

        # Get the bounding box
        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # Calculate bounding box size
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Calculate bounding box center
        x_center = xmin + box_width / 2
        y_center = ymin + box_height / 2

        # Normalize values between 0 and 1
        x_center = x_center / image_width
        y_center = y_center / image_height

        box_width = box_width / image_width
        box_height = box_height / image_height

        # Get class ID
        class_id = classes[class_name]

        # Create one YOLO label line
        yolo_line = (
            str(class_id)
            + " "
            + str(x_center)
            + " "
            + str(y_center)
            + " "
            + str(box_width)
            + " "
            + str(box_height)
        )

        yolo_labels.append(yolo_line)

    # Copy the image to the validation folder
    destination_image = (
        val_images_output
        + "/"
        + image_id
        + ".jpg"
    )

    shutil.copy(image_path, destination_image)

    # Create the corresponding YOLO label file
    destination_label = (
        val_labels_output
        + "/"
        + image_id
        + ".txt"
    )

    with open(destination_label, "w") as file:
        for label in yolo_labels:
            file.write(label + "\n")
