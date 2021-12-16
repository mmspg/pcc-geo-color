#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyntcloud import PyntCloud
import numpy as np
import os
from tqdm import tqdm
import argparse

def partition_pc(pc_file, block_size, keep_size, normalize, input_dir, output_dir, total_blocks, tqdm_handle):
    """Partition the PC and normalize color values.
    """
    pc = PyntCloud.from_file(os.path.join(input_dir, pc_file))
    points = pc.points

    max_range = max(points.x.max(), points.y.max(), points.z.max())
    depth = (np.floor(np.log2(max_range))+1)
    
    resolution = 2 ** depth
    
    steps = int(resolution / block_size)
    valid_blocks = 0
    
    for i in range(steps):
        for j in range(steps):
            for k in range(steps):
                tmp = points[
                    ((points.x >= (i * block_size)) & (points.x < ((i + 1) * block_size))) &
                    ((points.y >= (j * block_size)) & (points.y < ((j + 1) * block_size))) &
                    ((points.z >= (k * block_size)) & (points.z < ((k + 1) * block_size)))
                ]
                # save the block if it has enough points
                if tmp.shape[0] > keep_size:
                    # move coordinates back to [0, block_size]
                    tmp.x -= i * block_size
                    tmp.y -= j * block_size
                    tmp.z -= k * block_size
                    # normalize rgb values
                    if normalize:
                        tmp.red /= 255
                        tmp.green /= 255
                        tmp.blue /= 255
                    # save the block
                    new_pc = PyntCloud(tmp)
                    new_pc.to_file(os.path.join(output_dir, f'{pc_file[:-4]}_nor_i{i:02d}j{j:02d}k{k:02d}.ply'))
                    
                    valid_blocks += 1
                    total_blocks += 1
                    tqdm_handle.set_description(f'{valid_blocks} valid blocks / {total_blocks} in total')
                    
    return valid_blocks

def run(args):

    ply_files = sorted([f for f in os.listdir(args.input_dir) if '.ply' in f])
    print(f'There are {len(ply_files)} .ply files.')
        
    os.makedirs(args.output_dir, exist_ok=True)

    total_blocks = 0
    tqdm_handle = tqdm(ply_files)

    for pc_file in tqdm_handle:
        valid_block = partition_pc(pc_file, args.block_size, args.keep_size, True, args.input_dir, args.output_dir, total_blocks, tqdm_handle)
        total_blocks += valid_block

    print(f'There are {total_blocks} valid blocks in total.')


################################################################################
# Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='partition.py',
        description='Partition dataset into blocks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        help='Directory with point clouds to be partitioned.')
    parser.add_argument(
        'output_dir',
        help='Directory where to save point cloud blocks.')

    parser.add_argument(
        '--block_size',
        type=int, help='Block size of partition blocks.', default=32)
    parser.add_argument(
        '--keep_size', type=int, default=500,
        help='Minimum number of points accepted in each block.')

    args = parser.parse_args()

    run(args)
