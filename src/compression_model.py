#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tensorflow compression model setup
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

from collections import namedtuple
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_compression as tfc

from cnn_network import encoder_network, decoder_network
from color_loss import color_loss
from focal_loss import focal_loss


def pc_to_tf(points, dense_tensor_shape, task, channels_last=False):
    x = points

    paddings = [[0, 0], [1, 0]] if not channels_last else [[0, 0], [0, 1]]

    geo_indices = tf.pad(x[:, :3], paddings, constant_values=0)
    if task == 'geometry':
        indices = tf.cast(geo_indices, tf.int64)
        values = tf.ones_like(x[:, 0])
    else:
        r_indices = tf.pad(x[:, :3], paddings, constant_values=1)
        g_indices = tf.pad(x[:, :3], paddings, constant_values=2)
        b_indices = tf.pad(x[:, :3], paddings, constant_values=3)
        indices = tf.cast(tf.concat([geo_indices, r_indices, g_indices, b_indices], 0), tf.int64)
        values = tf.concat([tf.ones_like(x[:, 0]), x[:, 3], x[:, 4], x[:, 5]], 0)
    st = tf.sparse.SparseTensor(indices, values, dense_tensor_shape)

    return st


def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)

    return x


def quantize_tensor(x):
    x = tf.round(x)
    x = tf.cast(x, tf.uint8)

    return x


def input_fn(features, batch_size, dense_tensor_shape, preprocess_threads, task, channels_last=False, repeat=True, prefetch_size=1):
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        if task == 'geometry':
            dataset = tf.data.Dataset.from_generator(lambda: iter(
                features), tf.float32, tf.TensorShape([None, 3]))
        else:
            dataset = tf.data.Dataset.from_generator(lambda: iter(
                features), tf.float32, tf.TensorShape([None, 6]))
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape, task,
                                                 channels_last), num_parallel_calls=preprocess_threads)
        dataset = dataset.map(lambda x: (process_x(x, dense_tensor_shape),
                                         zero), num_parallel_calls=preprocess_threads)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_size)

    return dataset.make_one_shot_iterator().get_next()


def analysis_transform(tensor, num_filters, data_format):
    with tf.variable_scope('analysis'):
        tensor = encoder_network(tensor, num_filters, data_format)

    return tensor


def synthesis_transform(tensor, task, num_filters, data_format):
    with tf.variable_scope('synthesis'):
        tensor = decoder_network(tensor, task, num_filters, data_format)

    return tensor


def model_fn(features, labels, mode, params):
    if params.get('decompress') is None:
        params['decompress'] = False
    params = namedtuple('Struct', params.keys())(*params.values())
    del labels  # Unused
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    if params.decompress:
        assert mode == tf.estimator.ModeKeys.PREDICT, 'Decompression must use prediction mode'
        y_shape = params.y_shape
        y_shape = [params.num_filters] + [int(s) for s in y_shape]
        x_shape = tf.constant(params.x_shape, dtype=tf.int64)
        entropy_bottleneck = tfc.EntropyBottleneck(data_format=params.data_format, dtype=tf.float32)
        y_hat = entropy_bottleneck.decompress(features, y_shape, channels=params.num_filters)

        x_hat = synthesis_transform(y_hat, params.task, params.num_filters, params.data_format)

        if params.task == 'geometry':
            x_hat_quant = quantize_tensor(x_hat)
            predictions = {
                'x_hat': x_hat,
                'x_hat_quant': x_hat_quant
            }
        elif params.task == 'color':
            predictions = {
                'x_hat': x_hat
            }
        elif params.task == 'geometry+color':
            x_hat_quant = quantize_tensor(x_hat[:, 0])
            x_hat_quant = tf.expand_dims(x_hat_quant, 1)
            predictions = {
                'x_hat': x_hat,
                'x_hat_quant': x_hat_quant
            }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Get training patch from dataset
    x = features
    geo_x = x[:, 0] if not params.channels_last else x[:, :, :, :, 0]
    if params.task != 'geometry':
        color_x = x[:, 1:] if not params.channels_last else x[:, :, :, :, 1:]

    num_voxels = tf.cast(tf.size(geo_x), tf.float32)
    num_occupied_voxels = tf.reduce_sum(geo_x)

    # Build autoencoder
    y = analysis_transform(x, params.num_filters, params.data_format)
    entropy_bottleneck = tfc.EntropyBottleneck(data_format=params.data_format)
    y_tilde, likelihoods = entropy_bottleneck(y, training=training)
    x_tilde = synthesis_transform(y_tilde, params.task, params.num_filters, params.data_format)

    # Total number of bits divided by number of occupied pixels
    log_likelihoods = tf.math.log(likelihoods)
    train_mbpov = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_occupied_voxels)

    # For compression
    if mode == tf.estimator.ModeKeys.PREDICT:
        y_string = entropy_bottleneck.compress(y)

        # Remove batch and channels dimensions
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        batch_size = x_shape[0]

        # Repeat batch_size times
        def repeat(t, n):
            return tf.reshape(tf.tile(t, [n]), tf.concat([[n], tf.shape(t)], 0))
        if params.channels_last:
            x_shape_rep = repeat(x_shape[1:4], batch_size)
            y_shape_rep = repeat(y_shape[1:4], batch_size)
        else:
            x_shape_rep = repeat(x_shape[2:], batch_size)
            y_shape_rep = repeat(y_shape[2:], batch_size)

        predictions = {
            'y_string': y_string,
            'x_shape': x_shape_rep,
            'y_shape': y_shape_rep
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Calculate loss
    if params.task == 'geometry':
        train_focal = focal_loss(x, x_tilde, gamma=params.gamma,
                                 alpha=params.alpha) / num_voxels
        train_loss = params.lmbda_g * train_focal + train_mbpov
        tf.summary.scalar('focal_loss', train_focal)
    elif params.task == 'color':
        train_color = color_loss(color_x, x_tilde, geo_x,
                                 loss_type=params.loss_type,
                                 ssim_filter=params.ssim_filter,
                                 loss_weight=params.loss_weight,
                                 channels_last=params.channels_last) / num_occupied_voxels
        train_color = -tf.math.log(1 - train_color)
        train_loss = params.lmbda_c * train_color + train_mbpov
        tf.summary.scalar('color_loss', train_color)
    elif params.task == 'geometry+color':
        train_focal = focal_loss(geo_x, x_tilde[:, 0],
                                 gamma=params.gamma, alpha=params.alpha) / num_voxels
        train_color = color_loss(color_x, x_tilde[:, 1:], geo_x,
                                 loss_type=params.loss_type,
                                 ssim_filter=params.ssim_filter,
                                 loss_weight=params.loss_weight,
                                 channels_last=params.channels_last) / num_occupied_voxels
        train_color = -tf.math.log(1 - train_color)
        train_loss = params.lmbda_g * train_focal + params.lmbda_c * train_color + train_mbpov
        tf.summary.scalar('focal_loss', train_focal)
        tf.summary.scalar('color_loss', train_color)

    # Main metrics
    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("mbpov", train_mbpov)
    tf.summary.scalar("num_occupied_voxels", num_occupied_voxels)
    tf.summary.scalar("num_voxels", num_voxels)

    # Additional metrics
    if params.additional_metrics:
        train_bpv = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_voxels)

        tf.summary.histogram("y", y)
        tf.summary.histogram("y_tilde", y_tilde)
        tf.summary.histogram("likelihoods", likelihoods)
        tf.summary.histogram("log_likelihoods", log_likelihoods)

    # Evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=100,
            output_dir=os.path.join(params.checkpoint_dir, 'eval'),
            summary_op=tf.summary.merge_all())

        if params.additional_metrics:
            if params.task == 'geometry':
                logging_hook = tf.train.LoggingTensorHook({
                    'loss': train_loss,
                    'focal_loss': train_focal,
                    'mbpov': train_mbpov,
                    'ovoxels': num_occupied_voxels}, every_n_iter=100)
            elif params.task == 'color':
                logging_hook = tf.train.LoggingTensorHook({
                    'loss': train_loss,
                    'color_loss': train_color,
                    'mbpov': train_mbpov,
                    'ovoxels': num_occupied_voxels}, every_n_iter=100)
            elif params.task == 'geometry+color':
                logging_hook = tf.train.LoggingTensorHook({
                    'loss': train_loss,
                    'focal_loss': train_focal,
                    'color_loss': train_color,
                    'mbpov': train_mbpov,
                    'ovoxels': num_occupied_voxels}, every_n_iter=100)
            return tf.estimator.EstimatorSpec(mode,
                                              loss=train_loss,
                                              evaluation_hooks=[logging_hook, summary_hook])
        return tf.estimator.EstimatorSpec(mode,
                                          loss=train_loss,
                                          evaluation_hooks=[summary_hook])

    # Minimize loss and auxiliary loss, and execute update op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=tf.train.get_global_step())

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=os.path.join(params.checkpoint_dir, 'train'),
        summary_op=tf.summary.merge_all())

    if params.additional_metrics:
        if params.task == 'geometry':
            logging_hook = tf.train.LoggingTensorHook({
                'focal_loss': train_focal,
                'mbpov': train_mbpov,
                'ovoxels': num_occupied_voxels}, every_n_iter=100)
        elif params.task == 'color':
            logging_hook = tf.train.LoggingTensorHook({
                'color_loss': train_color,
                'mbpov': train_mbpov,
                'ovoxels': num_occupied_voxels}, every_n_iter=100)
        elif params.task == 'geometry+color':
            logging_hook = tf.train.LoggingTensorHook({
                'focal_loss': train_focal,
                'color_loss': train_color,
                'mbpov': train_mbpov,
                'ovoxels': num_occupied_voxels}, every_n_iter=100)

        return tf.estimator.EstimatorSpec(mode,
                                          loss=train_loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook, summary_hook])

    return tf.estimator.EstimatorSpec(mode,
                                      loss=train_loss,
                                      train_op=train_op,
                                      training_hooks=[summary_hook])
