#!/usr/bin/env python3
"""
    A module that defines a class Dataset
    and its instance based on data
"""


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    A class Dataset that loads and preps a dataset
    for machine translation with tensorflow
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en", split="train",
            as_supervised=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en", split="validation",
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=(2**15)
        )
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=(2**15)
        )
        return tokenizer_pt, tokenizer_en
