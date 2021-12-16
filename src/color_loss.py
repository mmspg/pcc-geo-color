#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Color loss implementation
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops


class Pos(tf.keras.constraints.Constraint):
    """Constrains the weights to be positive."""

    def __call__(self, w):
        return w * math_ops.cast(math_ops.greater(w, 0.), K.floatx())


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size // 2 + 1:size // 2 +
                                      1, -size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    z_data = np.expand_dims(z_data, axis=-1)
    z_data = np.expand_dims(z_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    z = tf.constant(x_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2)))

    return g / tf.reduce_sum(g)


def ssim_3d_simple(pc1, pc2, filter_size=11, channels_last=False):
    data_format = 'NCDHW' if not channels_last else 'NDHWC'
    pc1 = tf.expand_dims(pc1, 1)
    pc2 = tf.expand_dims(pc2, 1)
    windows = tf.ones([filter_size, filter_size, filter_size, 1, 1]) / (filter_size ** 3)

    mu1 = tf.nn.conv3d(pc1, windows, strides=[1, 1, 1, 1, 1],
                       padding='SAME', data_format=data_format)
    mu2 = tf.nn.conv3d(pc2, windows, strides=[1, 1, 1, 1, 1],
                       padding='SAME', data_format=data_format)
    mu1_sq = K.pow(mu1, 2)
    mu2_sq = K.pow(mu2, 2)

    sigma1_sq = tf.nn.conv3d(
        pc1 * pc1, windows, strides=[1, 1, 1, 1, 1], padding='SAME', data_format=data_format) - mu1_sq
    sigma2_sq = tf.nn.conv3d(
        pc2 * pc2, windows, strides=[1, 1, 1, 1, 1], padding='SAME', data_format=data_format) - mu2_sq

    # value = K.abs(sigma1_sq - sigma2_sq) / (K.maximum(K.abs(sigma1_sq), K.abs(sigma2_sq)) + K.epsilon())

    sigma1 = K.sqrt(sigma1_sq)
    sigma2 = K.sqrt(sigma2_sq)

    value_mean = K.abs(mu1 - mu2) / (K.maximum(K.abs(mu1), K.abs(mu2)) + K.epsilon())
    value_sigma = K.abs(sigma1 - sigma2) / (K.maximum(K.abs(sigma1), K.abs(sigma2)) + K.epsilon())
    value = tf.math.multiply(value_mean, value_sigma)

    return tf.squeeze(value)


def ssim_3d(pc1, pc2, filter_size=11, sigma=1.5, K1=0.01, K2=0.03, channels_last=False):
    data_format = 'NCDHW' if not channels_last else 'NDHWC'
    pc1 = tf.expand_dims(pc1, 1)
    pc2 = tf.expand_dims(pc2, 1)
    windows = _tf_fspecial_gauss(filter_size, sigma)

    C1 = K1 ** 2
    C2 = K2 ** 2

    mu1 = tf.nn.conv3d(pc1, windows, strides=[1, 1, 1, 1, 1],
                       padding='SAME', data_format=data_format)
    mu2 = tf.nn.conv3d(pc2, windows, strides=[1, 1, 1, 1, 1],
                       padding='SAME', data_format=data_format)
    mu1_sq = K.pow(mu1, 2)
    mu2_sq = K.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv3d(
        pc1 * pc1, windows, strides=[1, 1, 1, 1, 1], padding='SAME', data_format=data_format) - mu1_sq
    sigma2_sq = tf.nn.conv3d(
        pc2 * pc2, windows, strides=[1, 1, 1, 1, 1], padding='SAME', data_format=data_format) - mu2_sq
    sigma12 = tf.nn.conv3d(
        pc1 * pc2, windows, strides=[1, 1, 1, 1, 1], padding='SAME', data_format=data_format) - mu1_mu2

    value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return tf.squeeze(0.5 * (1 - value))


def color_loss(y_true, y_pred, g_true, loss_type='l2', ssim_filter=11, loss_weight='f,1,1,1', channels_last=False):
    t1 = y_true[:, 0] if not channels_last else y_true[:, :, :, :, 0]
    t2 = y_true[:, 1] if not channels_last else y_true[:, :, :, :, 1]
    t3 = y_true[:, 2] if not channels_last else y_true[:, :, :, :, 2]

    p1 = y_pred[:, 0] if not channels_last else y_pred[:, :, :, :, 0]
    p2 = y_pred[:, 1] if not channels_last else y_pred[:, :, :, :, 1]
    p3 = y_pred[:, 2] if not channels_last else y_pred[:, :, :, :, 2]

    p1 = tf.where(tf.equal(g_true, 1), p1, t1)
    p2 = tf.where(tf.equal(g_true, 1), p2, t2)
    p3 = tf.where(tf.equal(g_true, 1), p3, t3)

    if loss_type == 'l2':
        def loss_func(t, p): return K.pow(t - p, 2)
    elif loss_type == 'l1':
        def loss_func(t, p): return K.abs(t - p)
    elif loss_type == 'ssim_simple':
        def loss_func(t, p): return ssim_3d_simple(
            t, p, filter_size=ssim_filter, channels_last=channels_last)
    elif loss_type == 'ssim':
        def loss_func(t, p): return ssim_3d(
            t, p, filter_size=ssim_filter, channels_last=channels_last)

    loss_1 = loss_func(t1, p1)
    loss_2 = loss_func(t2, p2)
    loss_3 = loss_func(t3, p3)

    loss_1 = tf.where(tf.equal(g_true, 1), loss_1, tf.zeros_like(loss_1))
    loss_2 = tf.where(tf.equal(g_true, 1), loss_2, tf.zeros_like(loss_2))
    loss_3 = tf.where(tf.equal(g_true, 1), loss_3, tf.zeros_like(loss_3))

    if loss_weight.split(',')[0] == 'f':
        [w1, w2, w3] = [float(a) for a in loss_weight.split(',')[1:4]]
    elif loss_weight.split(',')[0] == 't':
        [w1, w2, w3] = [tf.Variable(float(loss_weight.split(',')[1]), trainable=True, name='w1', constraint=Pos()),
                        tf.Variable(float(loss_weight.split(',')[
                                    2]), trainable=True, name='w2', constraint=Pos()),
                        tf.Variable(float(loss_weight.split(',')[3]), trainable=True, name='w3', constraint=Pos())]
        tf.summary.scalar('w1', w1)
        tf.summary.scalar('w2', w2)
        tf.summary.scalar('w3', w3)
    loss_1 = w1 * loss_1
    loss_2 = w2 * loss_2
    loss_3 = w3 * loss_3

    return K.sum(loss_1 + loss_2 + loss_3) / (w1 + w2 + w3)
