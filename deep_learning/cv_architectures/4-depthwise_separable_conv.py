#!/usr/bin/env python3
"""This module implements a depthwise separable convolution block.
"""
from tensorflow import keras


def depthwise_separable_conv(X, filters, stride=1):
    """
    Implement a depthwise separable convolution block
    which is core building block of MobileNetV1.
    The block should consist of:
    - A depthwise convolution (DepthwiseConv2D) with a 3×3 kernel.
    - Batch Normalization and ReLU activation.
    - A pointwise convolution (1×1 Conv2D).
    - Batch Normalization and ReLU activation.

    Args:
        X: input tensor.
        filters: number of output channels for the pointwise convolution.
        stride: stride applied to the depthwise convolution.

    Returns:
        the output tensor of the depthwise separable convolution block.
    """
    x = keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=stride,
        padding='same',
        use_bias=False
    )(X)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        strides=1,
        padding='same',
        use_bias=False
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return x
