#!/usr/bin/env python3
"""
This module builds, trains, and saves an image classifier
using transfer learning on the Stanford Cars dataset.
"""
import tensorflow as tf
from tensorflow import keras


def load_datasets():
    """
    Load and split the Caltech-101 dataset.

    Returns:
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.
        num_classes: Number of classes.
    """
    # 1. Basic settings
    dataset_dir = "101_ObjectCategories"
    image_size = (224, 224)
    batch_size = 32
    seed = 42

    # 2. Load training dataset
    train_dataset = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    # 3. Load validation dataset
    validation_dataset = keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int"
    )

    num_classes = len(train_dataset.class_names)

    # 4. Improve dataset input performance
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(
        buffer_size=autotune
    )

    return train_dataset, validation_dataset, num_classes


def build_model(num_classes):
    """
    Build the transfer learning model.

    Args:
        num_classes: Number of output classes.

    Returns:
        model: Complete Keras model.
        base_model: Pretrained MobileNetV2 model.
    """
    # 1. Data augmentation
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip(
                "horizontal",
                seed=42
            ),
            keras.layers.RandomRotation(
                0.15,
                seed=42
            ),
            keras.layers.RandomZoom(
                0.15,
                seed=42
            ),
            keras.layers.RandomContrast(
                0.1,
                seed=42
            ),
        ],
        name="data_augmentation"
    )

    # 2. Load pretrained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # 3. Phase 1: freeze pretrained model
    base_model.trainable = False

    # 4. Build complete transfer-learning model
    inputs = keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = keras.layers.Dropout(
        0.2,
        name="dropout"
    )(x)

    x = keras.layers.Dense(
        128,
        activation="relu",
        name="classification_features"
    )(x)

    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="caltech101_transfer_model"
    )

    return model, base_model


def train_feature_extraction(model, train_dataset, validation_dataset):
    """
    Train only the classification head.

    Args:
        model: Complete Keras model.
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.

    Returns:
        None
    """
    # 1. Compile Phase 1
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # 2. Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )

    # 3. Phase 1 training
    print("\nPhase 1: Training classification head...\n")

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=15,
        callbacks=[
            early_stopping,
            reduce_lr
        ]
    )


def fine_tune_model(model, base_model, train_dataset, validation_dataset):
    """
    Fine-tune the top layers of the pretrained model.

    Args:
        model: Complete Keras model.
        base_model: MobileNetV2 base model.
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.

    Returns:
        None
    """
    # 1. Phase 2: fine-tuning
    print("\nPhase 2: Fine-tuning MobileNetV2...\n")

    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        ):
            layer.trainable = False

    # 2. Recompile after changing trainable layers
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    )

    fine_tune_reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )

    # 3. Fine-tune
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=[
            fine_tune_early_stopping,
            fine_tune_reduce_lr
        ]
    )


def train_transfer_model():
    """
    Run the complete transfer learning pipeline.

    Returns:
        None
    """
    # Final evaluation
    train_dataset, validation_dataset, num_classes = (load_datasets())

    model, base_model = build_model(num_classes)

    model.summary()
    train_feature_extraction(model, train_dataset, validation_dataset)

    fine_tune_model(model, base_model, train_dataset, validation_dataset)

    validation_loss, validation_accuracy = model.evaluate(
        validation_dataset
    )

    print(
        "\nFinal validation accuracy:",
        f"{validation_accuracy * 100:.2f}%"
    )

    # Save trained model
    model.save("caltech101_model.h5")

    print("\nModel saved as caltech101_model.h5")


if __name__ == "__main__":
    train_transfer_model()
