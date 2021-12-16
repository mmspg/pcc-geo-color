#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Handle point clouds I/O
Code adpated from https://github.com/mauriceqch/pcc_geo_cnn
"""

import functools
from glob import glob
import logging
import multiprocessing

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from tqdm import tqdm


logger = logging.getLogger(__name__)


class PC:
    def __init__(self, points, p_min, p_max):
        self.points = points
        self.p_max = p_max
        self.p_min = p_min
        self.data = {}

        assert np.all(p_min < p_max), f"p_min <= p_max must be true : p_min {p_min}, p_max {p_max}"
        assert np.all(points[:, :3] < p_max), f"points must be inferior to p_max {p_max}"
        assert np.all(points[:, :3] >= p_min), f"points must be superior to p_min {p_min}"

    def __repr__(self):
        return f"<PC with {self.points.shape[0]} points (p_min: {self.p_min}, p_max: {self.p_max})>"

    def is_empty(self):
        return self.points.shape[0] == 0

    def p_mid(self):
        p_min = self.p_min
        p_max = self.p_max
        return p_min + ((p_max - p_min) / 2.)


def df_to_pc(df, p_min, p_max, task):
    if task == 'geometry':
        points = df[['x', 'y', 'z']].values
    else:
        points = df[['x', 'y', 'z', 'red', 'green', 'blue']].values
    return PC(points, p_min, p_max)


def pa_to_df(points, task):
    if task == 'geometry':
        if len(points) == 0:
            df = pd.DataFrame(
                data={
                    'x': [],
                    'y': [],
                    'z': []})
        else:
            df = pd.DataFrame(
                data={
                    'x': points[:, 0],
                    'y': points[:, 1],
                    'z': points[:, 2]}, dtype=np.float32)
    else:
        if len(points) == 0:
            df = pd.DataFrame(
                data={
                    'x': [],
                    'y': [],
                    'z': [],
                    'red': [],
                    'green': [],
                    'blue': []})
        else:
            df = pd.DataFrame(
                data={
                    'x': points[:, 0],
                    'y': points[:, 1],
                    'z': points[:, 2],
                    'red': points[:, 3],
                    'green': points[:, 4],
                    'blue': points[:, 5]}, dtype=np.float32)

    return df


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def load_pc(path, p_min, p_max, task):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points, p_min, p_max, task)
    logger.debug(f"Loaded PC {path}")
    return ret


def write_df(path, df):
    pc = PyntCloud(df)
    pc.to_file(path)


def get_shape_data(task, resolution, channels_last):
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    if task == 'geometry':
        dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')
    else:
        dense_tensor_shape = np.concatenate([[4], p_max]).astype('int64')

    if channels_last:
        dense_tensor_shape = dense_tensor_shape[[1, 2, 3, 0]]

    return p_min, p_max, dense_tensor_shape


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_points_func(x, p_min, p_max, task):
    return load_pc(x, p_min, p_max, task).points


def load_points(files, p_min, p_max, task, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        f = functools.partial(load_points_func, p_min=p_min, p_max=p_max, task=task)
        points = np.array(list(tqdm(p.imap(f, files, batch_size), total=files_len)))

    return points
