#!/usr/bin/env python3
"""This module builds the feature extraction backbone of MobileNetV1.
"""
from tensorflow import keras


def depthwise_block(x, filters, strides=(1, 1)):
    """Builds a Depthwise Separable Convolution block.

    Consists of:
      1. DepthwiseConv2D (3x3) + BatchNormalization + ReLU
      2. Pointwise Conv2D (1x1) + BatchNormalization + ReLU
    """
    # 1. Spatial filtering (Depthwise)
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), strides=strides, padding="same", use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # 2. Channel combination (Pointwise 1x1)
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x


def mobilenet_backbone(inputs):
    """
    Build the feature extraction backbone of MobileNetV1.
    The backbone should:
    - Begin with a standard 3×3 convolution with stride 2.
    - Stack multiple depthwise separable convolution blocks.
    - Perform spatial downsampling by increasing stride at specific stages.
    - Follow the original MobileNetV1 architectural pattern.

    Args:
        inputs: input tensor to the network.

    Returns:
        the output tensor of the MobileNet backbone (before classification).
    """
    # 1. Entry Convolution Layer
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False,
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    # 2. Depthwise Separable Blocks
    x = depthwise_block(x, filters=64, strides=(1, 1))

    # Downsample to 56x56
    x = depthwise_block(x, filters=128, strides=(2, 2))
    x = depthwise_block(x, filters=128, strides=(1, 1))

    # Downsample to 28x28
    x = depthwise_block(x, filters=256, strides=(2, 2))
    x = depthwise_block(x, filters=256, strides=(1, 1))

    # Downsample to 14x14
    x = depthwise_block(x, filters=512, strides=(2, 2))
    # 5 repeated blocks with 512 filters and stride 1
    for _ in range(5):
        x = depthwise_block(x, filters=512, strides=(1, 1))

    # Downsample to 7x7
    x = depthwise_block(x, filters=1024, strides=(2, 2))
    x = depthwise_block(x, filters=1024, strides=(1, 1))

    return x
