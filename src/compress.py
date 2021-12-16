#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compression script
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import argparse
import gzip
import logging
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import compression_model
import pc_io


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)
tf.set_random_seed(42)


TYPE = np.uint16
DTYPE = np.dtype(np.uint16)
SHAPE_LEN = 3


def compress(nn_output):
    x_shape = nn_output['x_shape']
    y_shape = nn_output['y_shape']
    y_string = nn_output['y_string']
    x_shape_b = np.array(x_shape, dtype=TYPE).tobytes()
    y_shape_b = np.array(y_shape, dtype=TYPE).tobytes()
    representation = x_shape_b + y_shape_b + y_string

    return representation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        help='Input directory.')
    parser.add_argument(
        'input_pattern',
        help='Mesh detection pattern.')
    parser.add_argument(
        'output_dir',
        help='Output directory.')
    parser.add_argument(
        'checkpoint_dir',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size.')
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=64)

    parser.add_argument(
        '--task', type=str, default='color',
        help='Compression tasks (geometry/color/geometry+color).')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--color_space', type=str, default='rgb',
        help='Color space type.')
    parser.add_argument(
        '--network_type', type=str, default='base',
        help='Neural network type.')
    parser.add_argument(
        '--channels_last', action='store_true',
        help='Use channels last instead of channels first.')

    args = parser.parse_args()

    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'

    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)

    files = pc_io.get_files(input_glob)
    assert len(files) > 0, "No input files found"
    filenames = [x[len_input_dir + 1:] for x in files]
    output_files = [os.path.join(args.output_dir, x + '.bin') for x in filenames]

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(
        args.task, args.resolution, args.channels_last)
    points = pc_io.load_points(files, p_min, p_max, args.task, batch_size=args.read_batch_size)

    estimator = tf.estimator.Estimator(
        model_fn=compression_model.model_fn,
        model_dir=args.checkpoint_dir,
        params={
            'task': args.task,
            'num_filters': args.num_filters,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT,
            'color_space': args.color_space,
            'channels_last': args.channels_last,
            'network_type': args.network_type,
            'batch_size': args.batch_size})

    result = estimator.predict(
        input_fn=lambda: compression_model.input_fn(
            points, args.batch_size, dense_tensor_shape, args.preprocess_threads, args.task, args.channels_last, repeat=False),
        predict_keys=['y_string', 'x_shape', 'y_shape'])

    for ret, ori_file, output_file in zip(result, tqdm(files), output_files):
        logger.info(f'Writing {ori_file} to {output_file}')
        output_dir, _ = os.path.split(output_file)
        os.makedirs(output_dir, exist_ok=True)
        with gzip.open(output_file, "wb") as f:
            representation = compress(ret)
            f.write(representation)
