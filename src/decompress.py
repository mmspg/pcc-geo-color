#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decompression script
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import argparse
import gzip
import logging
import multiprocessing
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
DTYPE = np.dtype(TYPE)
SHAPE_LEN = 3


def read_from_buffer(f, n, dtype):
    return np.frombuffer(f.read(int(np.dtype(dtype).itemsize * n)), dtype=dtype)


def load_compressed_file(c_file):
    with gzip.open(c_file, "rb") as f:
        x_shape = read_from_buffer(f, SHAPE_LEN, np.uint16)
        y_shape = read_from_buffer(f, SHAPE_LEN, np.uint16)
        string = f.read()
        return x_shape, y_shape, string


def load_compressed_files(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading data into memory (parallel reading)')
        data = np.array(
            list(tqdm(p.imap(load_compressed_file, files, batch_size), total=files_len)))

    return data


def input_fn(features, batch_size):
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        dataset = tf.data.Dataset.from_generator(lambda: features, (tf.string))
        dataset = dataset.map(lambda t: (t, zero))
        dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'geo_dir',
        help='Geometry input directory.')
    parser.add_argument(
        'geo_pattern',
        help='Geometry mesh detection pattern.')
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
        '--output_extension', default='.ply',
        help='Output extension.')
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

    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first' if not args.channels_last else 'channels_last'

    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)
    files = pc_io.get_files(input_glob)
    assert len(files) > 0, "No input files found"

    if args.task == 'color':
        args.geo_dir = os.path.normpath(args.geo_dir)
        len_geo_dir = len(args.geo_dir)
        assert os.path.exists(args.geo_dir), "Geometry input directory not found"

        geo_glob = os.path.join(args.geo_dir, args.geo_pattern)
        geo_files = pc_io.get_files(geo_glob)
        assert len(geo_files) > 0, "No geometry input files found"

        p_min, p_max, _ = pc_io.get_shape_data(args.task, args.resolution, args.channels_last)
        points = pc_io.load_points(geo_files, p_min, p_max, args.task)

    filenames = [x[len_input_dir + 1:] for x in files]
    output_files = [os.path.join(args.output_dir, x + '.ply') for x in filenames]

    compressed_data = load_compressed_files(files, args.read_batch_size)
    x_shape = compressed_data[0][0]
    y_shape = compressed_data[0][1]
    assert np.all([np.all(x[0] == x_shape) for x in compressed_data]), 'All x_shape must be equal'
    assert np.all([np.all(x[1] == y_shape) for x in compressed_data]), 'All y_shape must be equal'
    compressed_strings = (x[2] for x in compressed_data)

    estimator = tf.estimator.Estimator(
        model_fn=compression_model.model_fn,
        model_dir=args.checkpoint_dir,
        params={
            'task': args.task,
            'num_filters': args.num_filters,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT,
            'decompress': True,
            'x_shape': x_shape,
            'y_shape': y_shape,
            'color_space': args.color_space,
            'network_type': args.network_type,
            'channels_last': args.channels_last})

    if args.task == 'color':
        result = estimator.predict(
            input_fn=lambda: input_fn(compressed_strings, args.batch_size),
            predict_keys=['x_hat'])
    else:
        result = estimator.predict(
            input_fn=lambda: input_fn(compressed_strings, args.batch_size),
            predict_keys=['x_hat', 'x_hat_quant'])

    len_files = len(files)
    i = 0
    if args.task == 'geometry':
        for ret, ori_file, output_file in zip(result, files, output_files):
            logger.info(f'{i + 1}/{len_files} - Writing {ori_file} to {output_file}')
            output_dir, _ = os.path.split(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Remove the geometry channel
            pa = np.argwhere(ret['x_hat_quant'][0]).astype('float32')
            pc_io.write_df(output_file, pc_io.pa_to_df(pa, args.task))
            i += 1
    elif args.task == 'color':
        for ret, ori_file, point, output_file in zip(result, files, points, output_files):
            logger.info(f'{i + 1}/{len_files} - Writing {ori_file} to {output_file}')
            output_dir, _ = os.path.split(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # get geometry from the input file
            pa = point[:, :3]
            color_pa = []
            for p in pa:
                pos = tuple(p.astype(int))
                if not args.channels_last:
                    tmp = [*p, ret['x_hat'][0][pos], ret['x_hat'][1][pos], ret['x_hat'][2][pos]]
                else:
                    tmp = [*p, ret['x_hat'][:, :, :, 0][pos], ret['x_hat']
                           [:, :, :, 1][pos], ret['x_hat'][:, :, :, 2][pos]]
                color_pa.append(tmp)

            color_pa = np.array(color_pa)
            pc_io.write_df(output_file, pc_io.pa_to_df(color_pa, args.task))
            i += 1
    elif args.task == 'geometry+color':
        for ret, ori_file, output_file in zip(result, files, output_files):
            logger.info(f'{i + 1}/{len_files} - Writing {ori_file} to {output_file}')
            output_dir, _ = os.path.split(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Remove the geometry channel
            pa = np.argwhere(ret['x_hat_quant'][0]).astype('float32') if not args.channels_last else np.argwhere(
                ret['x_hat_quant'][:, :, :, 0]).astype('float32')
            color_pa = []
            for p in pa:
                pos = tuple(p.astype(int))
                if not args.channels_last:
                    tmp = [*p, ret['x_hat'][1][pos], ret['x_hat'][2][pos], ret['x_hat'][3][pos]]
                else:
                    tmp = [*p, ret['x_hat'][:, :, :, 1][pos], ret['x_hat']
                           [:, :, :, 2][pos], ret['x_hat'][:, :, :, 3][pos]]
                color_pa.append(tmp)

            color_pa = np.array(color_pa)
            pc_io.write_df(output_file, pc_io.pa_to_df(color_pa, args.task))
            i += 1
