#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import os
import shutil
import argparse
import glob

def run(args):

    set_size = 10000

    random.seed(99)

    os.makedirs(args.output_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.input_dir, '*.ply'))

    indices = random.sample(range(len(files)), set_size)

    for index in indices:
        file_in = files[index]
        file_out = file_in.replace(args.input_dir, args.output_dir)
        shutil.copyfile(file_in, file_out)


################################################################################
# Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sample_dataset.py',
        description='Sample random blocks from dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        help='Directory with point clouds blocks to be sampled.')
    parser.add_argument(
        'output_dir',
        help='Directory where to save sampled point cloud blocks.')

    parser.add_argument(
        '--set_size',
        type=int, help='Number of point cloud blocks to be sampled.', default=10000)


    args = parser.parse_args()

    run(args)