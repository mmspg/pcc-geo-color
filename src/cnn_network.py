#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Encoder and decoder specs
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import tensorflow as tf


def encoder_network(tensor, num_filters, data_format):
    shared_params = {
        'filters': num_filters,
        'strides': (2, 2, 2),
        'padding': 'same',
        'kernel_initializer': 'he_uniform',
        'activation': tf.nn.relu,
        'data_format': data_format
    }
    layers = [
        tf.layers.Conv3D(kernel_size=(9, 9, 9), use_bias=True, **shared_params),
        tf.layers.Conv3D(kernel_size=(5, 5, 5), use_bias=True, **shared_params),
        tf.layers.Conv3D(filters=num_filters, kernel_size=(5, 5, 5), padding='same',
                         strides=(2, 2, 2), use_bias=False, kernel_initializer='he_uniform',
                         activation=tf.nn.relu, data_format=data_format)
    ]
    for layer in layers:
        tensor = layer(tensor)

    return tensor


def decoder_network(tensor, task, num_filters, data_format):
    if task == 'geometry':
        ch = 1
    elif task == 'color':
        ch = 3
    elif task == 'geometry+color':
        ch = 4

    shared_params = {
        'strides': (2, 2, 2),
        'padding': 'same',
        'use_bias': True,
        'data_format': data_format
    }
    layers = [
        tf.layers.Conv3DTranspose(filters=num_filters, kernel_size=(5, 5, 5),
                                  kernel_initializer='he_uniform',
                                  activation=tf.nn.relu, **shared_params),
        tf.layers.Conv3DTranspose(filters=num_filters, kernel_size=(5, 5, 5),
                                  kernel_initializer='he_uniform',
                                  activation=tf.nn.relu, **shared_params),
        tf.layers.Conv3DTranspose(filters=ch, kernel_size=(9, 9, 9),
                                  kernel_initializer='glorot_uniform',
                                  activation=tf.nn.sigmoid, **shared_params)
    ]
    for layer in layers:
        tensor = layer(tensor)

    return tensor
