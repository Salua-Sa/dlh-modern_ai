#!/usr/bin/env python3
"""This module implements a ResNet bottleneck residual block.
"""
from tensorflow import keras


def bottleneck_block(x, filters, stride=1, downsample=False, name=None):
    """
    Implements a ResNet bottleneck residual block.
    The block should consist of:
    - A 1×1 convolution that reduces the number of channels.
    - A 3×3 convolution applied to the reduced representation.
    - A 1×1 convolution that expands the channels by a factor of 4.
    - Batch Normalization after each convolution.
    - ReLU activation after the first and second convolutions.
    - A residual (skip) connection:
        - Identity shortcut if downsample=False.
        - Projection shortcut (1×1 convolution + BatchNorm) if downsample=True.
        - A final ReLU activation after adding the shortcut.

    Args:
        x: input tensor.
        filters: number of filters for the 3×3 convolution.
        stride: stride for the first convolution.
        downsample: boolean indicating whether to apply a projection shortcut.
        name: optional string to name the block layers.

    Returns:
        the output tensor of the bottleneck residual block.
    """
    # Keep the original input for the shortcut connection.
    shortcut = x

    # First 1*1 convolution, readuces the number of channels
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{name}_conv1' if name else None)(x)

    x = keras.layers.BatchNormalization(
        name=f'{name}_bn1' if name else None)(x)

    x = keras.layers.ReLU(
        name=f'{name}_relu1' if name else None)(x)

    # 3 * 3 convolution, extracts spatial features
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        name=f'{name}_conv2' if name else None)(x)

    x = keras.layers.BatchNormalization(
        name=f'{name}_bn2' if name else None)(x)

    x = keras.layers.ReLU(
        name=f'{name}_relu2' if name else None)(x)

    # Final 1*1 convolution, expands channels by a factor of 4
    x = keras.layers.Conv2D(
        filters=filters * 4,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False,
        name=f'{name}_conv3' if name else None)(x)

    x = keras.layers.BatchNormalization(
        name=f'{name}_bn3' if name else None)(x)

    # Projection shortcut, used when dimensions must change
    if downsample:
        shortcut = keras.layers.Conv2D(
            filters=filters * 4,
            kernel_size=(1, 1),
            strides=stride,
            padding='same',
            name=f'{name}_shortcut_conv' if name else None)(shortcut)

        shortcut = keras.layers.BatchNormalization(
            name=f'{name}_shortcut_bn' if name else None)(shortcut)

    # Add the main path and shortcut
    x = keras.layers.Add(
        name=f'{name}_add' if name else None)([x, shortcut])

    # Final activation
    x = keras.layers.ReLU(
        name=f'{name}_out' if name else None)(x)

    return x
