#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import argparse
import logging
import os
import random
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import compression_model
import pc_io


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
RANDOM_SEED = 42
set_seeds(RANDOM_SEED)


def train():
    """Train the model."""
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        args.task, args.resolution, args.channels_last)
    files = pc_io.get_files(args.train_glob)
    perm = np.random.permutation(len(files))
    points = pc_io.load_points(files[perm][:args.num_data], p_min, p_max, args.task)

    points_train = points[:-args.num_val]
    points_val = points[-args.num_val:]

    config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=1,
        save_checkpoints_steps=args.save_checkpoints_steps,
        keep_checkpoint_max=args.keep_checkpoint_max,
        log_step_count_steps=args.log_step_count_steps,
        save_summary_steps=args.save_summary_steps,
        tf_random_seed=RANDOM_SEED)
    estimator = tf.estimator.Estimator(
        model_fn=compression_model.model_fn,
        model_dir=args.checkpoint_dir,
        config=config,
        params={
            'task': args.task,
            'num_filters': args.num_filters,
            'alpha': args.alpha,
            'gamma': args.gamma,
            'lmbda_g': args.lmbda_g,
            'lmbda_c': args.lmbda_c,
            'additional_metrics': not args.no_additional_metrics,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT,
            'channels_last': args.channels_last,
            'ssim_filter': args.ssim_filter,
            'loss_type': args.loss_type,
            'loss_weight': args.loss_weight,
            'batch_size': args.batch_size})

    hooks = None
    if args.debug_address is not None:
        hooks = [tf_debug.TensorBoardDebugHook(args.debug_address)]

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: compression_model.input_fn(points_train, args.batch_size, dense_tensor_shape,
                                                    args.preprocess_threads, args.task, args.channels_last, prefetch_size=args.prefetch_size),
        max_steps=args.max_steps,
        hooks=hooks)
    val_spec = tf.estimator.EvalSpec(
        input_fn=lambda: compression_model.input_fn(points_val, args.batch_size, dense_tensor_shape,
                                                    args.preprocess_threads, args.task, args.channels_last, repeat=False, prefetch_size=32),
        steps=args.num_val // args.batch_size,
        throttle_secs=1,
        hooks=hooks)

    tf.estimator.train_and_evaluate(estimator, train_spec, val_spec)


################################################################################
# Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'train_glob',
        help='Glob pattern for identifying training data.')
    parser.add_argument(
        'checkpoint_dir',
        help='Directory where to save/load model checkpoints.')

    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=32)
    parser.add_argument(
        '--num_data', type=int, default=None,
        help='Number of total data we want to use (-1: use all data).')
    parser.add_argument(
        '--num_val', type=int, default=64,
        help='Number of validation data we want to use')

    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Report bitrate and distortion when training.')
    parser.add_argument(
        '--no_additional_metrics', action='store_true',
        help='Report additional metrics when training.')
    parser.add_argument(
        '--save_checkpoints_steps', type=int, default=1000,
        help='Save checkpoints every n steps during training.')
    parser.add_argument(
        '--keep_checkpoint_max', type=int, default=1,
        help='Maximum number of checkpoint files to keep.')
    parser.add_argument(
        '--log_step_count_steps', type=int, default=100,
        help='Log global step and loss every n steps.')
    parser.add_argument(
        '--save_summary_steps', type=int, default=100,
        help='Save summaries every n steps.')
    parser.add_argument(
        '--debug_address', default=None,
        help='TensorBoard debug address.')

    parser.add_argument(
        '--task', type=str, default='color',
        help='Compression tasks (geometry/color/geometry+color).')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for training.')
    parser.add_argument(
        '--prefetch_size', type=int, default=128,
        help='Number of batches to prefetch for training.')
    parser.add_argument(
        '--lmbda_g', type=float, default=400,
        help='Lambda (geometry) for rate-distortion tradeoff.')
    parser.add_argument(
        '--lmbda_c', type=float, default=400,
        help='Lambda (color) for rate-distortion tradeoff.')
    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help='Focal loss alpha.')
    parser.add_argument(
        '--gamma', type=float, default=2.0,
        help='Focal loss gamma.')
    parser.add_argument(
        '--max_steps', type=int, default=100000,
        help='Train up to this number of steps.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--loss_type', type=str, default='l2',
        help='Color loss type.')
    parser.add_argument(
        '--ssim_filter', type=int, default=11,
        help='Filter size for ssim loss.')
    parser.add_argument(
        '--loss_weight', type=str, default='f,1,1,1',
        help='Loss weights for different channels.')
    parser.add_argument(
        '--channels_last', action='store_true',
        help='Use channels last instead of channels first.')

    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'

    train()
