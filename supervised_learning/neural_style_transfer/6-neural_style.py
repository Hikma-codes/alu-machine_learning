#!/usr/bin/env python3
"""
Defines class NST that performs tasks for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer class

        Args:
            style_image: numpy.ndarray of shape (h, w, 3)
            content_image: numpy.ndarray of shape (h, w, 3)
            alpha: weight for content cost
            beta: weight for style cost
        """

        if not isinstance(style_image, np.ndarray) or \
                len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
                len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        style_h, style_w, style_c = style_image.shape
        content_h, content_w, content_c = content_image.shape

        if style_h <= 0 or style_w <= 0 or style_c != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if content_h <= 0 or content_w <= 0 or content_c != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that:
        - pixels are between 0 and 1
        - largest side is 512 pixels
        - aspect ratio is maintained
        - bicubic interpolation is used

        Returns:
            TensorFlow tensor of shape (1, h_new, w_new, 3)
        """

        if not isinstance(image, np.ndarray) or \
                len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, c = image.shape

        if h <= 0 or w <= 0 or c != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)

        resized = tf.image.resize(
            image,
            size=(h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )

        resized = resized / 255.0
        resized = tf.clip_by_value(resized, 0.0, 1.0)

        return resized

    def load_model(self):
        """
        Creates the model used to calculate the style and content costs.

        The model:
        - uses VGG19 as its base
        - does not include the top classification layer
        - uses ImageNet weights
        - outputs the required style and content layers
        - freezes all layers
        """

        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = []

        for layer in self.style_layers:
            outputs.append(vgg.get_layer(layer).output)

        outputs.append(vgg.get_layer(self.content_layer).output)

        self.model = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs
        )

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a tensor.

        Args:
            input_layer: tensor of shape (1, h, w, c)

        Returns:
            Gram matrix of shape (1, c, c)
        """

        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        if h is None or w is None or c is None:
            raise TypeError("input_layer must have a defined shape")

        features = tf.reshape(
            input_layer,
            (int(h * w), int(c))
        )

        gram = tf.matmul(
            features,
            features,
            transpose_a=True
        )

        gram = tf.expand_dims(gram, axis=0)

        gram /= tf.cast(h * w, tf.float32)

        return gram

    def generate_features(self):
        """
        Extracts the features used to calculate neural style cost.
        """

        vgg19 = tf.keras.applications.vgg19

        style_input = vgg19.preprocess_input(
            self.style_image * 255.0
        )

        content_input = vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(style_input)

        content_outputs = self.model(content_input)

        style_features = style_outputs[:-1]
        content_feature = content_outputs[-1]

        gram_style_features = []

        for feature in style_features:
            gram_style_features.append(
                self.gram_matrix(feature)
            )

        self.gram_style_features = gram_style_features
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.

        Args:
            style_output: tensor of shape (1, h, w, c)
            gram_target: tensor of shape (1, c, c)

        Returns:
            Style cost for the layer
        """

        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, h, w, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                len(gram_target.shape) != 3 or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]"
                .format(c, c)
            )

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(
            tf.square(gram_style - gram_target)
        )

    def style_cost(self, style_outputs):
        """
        Calculates the style cost for a generated image.

        Args:
            style_outputs: list containing the outputs of the style layers

        Returns:
            Style cost
        """

        if not isinstance(style_outputs, list) or \
                len(style_outputs) != len(self.style_layers):
            raise TypeError(
                "style_outputs must be a list with a length of {}"
                .format(len(self.style_layers))
            )

        weight = 1 / len(self.style_layers)
        style_cost = 0.0

        for i in range(len(self.style_layers)):
            style_cost += weight * self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )

        return style_cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for a generated image.

        Args:
            content_output: tensor of shape (1, h, w, c)

        Returns:
            Content cost
        """

        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                len(content_output.shape) != 4:
            raise TypeError(
                "content_output must be a tensor of rank 4"
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )
