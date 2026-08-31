#!/usr/bin/env python3
"""This module  builds the ResNet‑101 architecture.
"""
from tensorflow import keras
bottleneck_block = __import__('2-bottleneck_block').bottleneck_block


def make_layer(x, blocks, filters, stride, name=None):
    """Stack a sequence of bottleneck residual blocks into one stage.

    The first block applies the given stride and a projection
    shortcut (to both downsample spatially, when stride > 1, and to
    match the expanded channel count). Every subsequent block uses a
    stride of 1 and an identity shortcut.

    Args:
        x (tf.Tensor): input tensor.
        blocks (int): number of bottleneck blocks in this stage.
        filters (int): number of filters for the 3x3 convolution in
            each block (the stage's output channels are filters * 4).
        stride (int): stride applied by the stage's first block.
            Defaults to 1.
        name (str): prefix used to name the stage's blocks. Defaults
            to None.

    Returns:
        tf.Tensor: the output tensor of the stage.
    """
    x = bottleneck_block(
        x, filters, stride=stride, downsample=True,
        name=f'{name}_block1')
    for i in range(1, blocks):
        x = bottleneck_block(
            x, filters, stride=1, downsample=False,
            name=f'{name}_block{i + 1}')
    return x


def build_resnet101(input_shape=(224, 224, 3), num_classes=1000):
    """
    that builds the ResNet‑101 architecture that builds
    the ResNet‑101 architecture as described in
    Deep Residual Learning for Image Recognition” (2015).
    The architecture should:
    - Begin with an initial convolutional layer and max pooling.
    - Stack bottleneck residual blocks
    using the standard ResNet‑101 configuration:
            - 3 blocks in conv2_x
            - 4 blocks in conv3_x
            - 23 blocks in conv4_x
            - 3 blocks in conv5_x
    - Downsample spatial dimensions at the start of each stage
    (except the first).
    - End with global average pooling and a fully connected
    classification layer.

    Args:
        input_shape: tuple representing the input image shape.
        num_classes: number of output classes.

    Returns:
        the Keras model implementing the ResNet‑101 architecture.
    """
    inputs = K.Input(shape=input_shape, name="input_layer")

    # Initial conv + BN + ReLU + MaxPool
    x = keras.layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same',
        use_bias=False, name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2,
                                  padding='same', name='maxpool')(x)

    # conv2_x: 3 blocks, filters=64, stride=1
    x = make_layer(x, blocks=3, filters=64, stride=1, name='layer1')

    # conv3_x: 4 blocks, filters=128, stride=2
    x = make_layer(x, blocks=4, filters=128, stride=2, name='layer2')

    # conv4_x: 23 blocks, filters=256, stride=2
    x = make_layer(x, blocks=23, filters=256, stride=2, name='layer3')

    # conv5_x: 3 blocks, filters=512, stride=2
    x = make_layer(x, blocks=3, filters=512, stride=2, name='layer4')

    # Global average pooling + FC
    x = keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    outputs = keras.layers.Dense(num_classes, name='fc')(x)

    model = K.Model(inputs=inputs, outputs=outputs, name='resnet101')
    return model
