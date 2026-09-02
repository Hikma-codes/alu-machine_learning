#!/usr/bin/env python3
# Defines class NST that performs tasks for neural style transfer

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for Neural Style Transfer

    public class attributes:
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']
        content_layer = 'block5_conv2'

    instance attributes:
        style_image: preprocessed style image
        content_image: preprocessed style image
        alpha: weight for content cost
        beta: weight for style cost
        model: the Keras model used to calculate cost
        gram_style_features: list of gram matrices from style layer outputs
        content_feature: the content layer output of the content image

    class constructor:
        def __init__(self, style_image, content_image, alpha=1e4, beta=1)

    static methods:
        def scale_image(image):
            rescales an image so the pixel values are between 0 and 1
                and the largest side is 512 pixels
        def gram_matrix(input_layer):
            calculates gram matrices

    public instance methods:
        def load_model(self):
            creates model used to calculate cost from VGG19 Keras base model
        def generate_features(self):
            extracts the features used to calculate neural style cost
        def layer_style_cost(self, style_output, gram_target):
            calculates the style cost for a single layer
        def style_cost(self, style_outputs):
            calculates the style cost for generated image
        def content_cost(self, content_output):
            calculates the content cost for the generated image
        def total_cost(self, generated_image):
            calculates the total cost for the generated image
        def compute_grads(self, generated_image):
            calculates the gradients for the generated image
        def generate_image(self, iterations=1000, step=None, lr=0.01,
            beta1=0.9, beta2=0.99):
            generates the neural style transferred image
    """
