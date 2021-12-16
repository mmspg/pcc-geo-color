#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyntcloud import PyntCloud
from tqdm import tqdm
import pandas as pd
import os
import argparse

def merge_pc(ori_file, div_files, block_size, div_dir, output_dir, task):
    cur_div_files = [f for f in div_files if ori_file[:-4] in f]
    
    total_pieces = len(cur_div_files)
    if task == 'geometry':
        points = pd.DataFrame(data={ 'x': [], 'y': [], 'z': [] })
    else:
        points = pd.DataFrame(data={ 'x': [], 'y': [], 'z': [], 'red': [], 'green': [], 'blue': [] })
    
    for div_file in cur_div_files:
        div_pc = PyntCloud.from_file(os.path.join(div_dir, div_file))
        div_pc_points = div_pc.points
        ind = [int(div_file.split('_')[-1][1:3]), int(div_file.split('_')[-1][4:6]), int(div_file.split('_')[-1][7:9])]
        div_pc_points.x += ind[0] * block_size
        div_pc_points.y += ind[1] * block_size
        div_pc_points.z += ind[2] * block_size
        
        points = pd.concat([points, div_pc_points])
        
    points.reset_index(drop=True, inplace=True)
    if task != 'geometry':
        points['red'] = (points.red * 255).astype('uint8')
        points['green'] = (points.green * 255).astype('uint8')
        points['blue'] = (points.blue * 255).astype('uint8')
    res_pc = PyntCloud(points)
    res_pc.to_file(os.path.join(output_dir, f'{ori_file[:-4]}_dec.ply'))


def run(args):

    ori_files = sorted([f for f in os.listdir(args.ori_dir) if '.ply' in f])
    print(f'There are {len(ori_files)} .ply files.')

    div_files = sorted([f for f in os.listdir(args.div_dir) if '.ply' in f])
    print(f'There are {len(div_files)} divided .ply files.')

    os.makedirs(args.output_dir, exist_ok=True)
    tqdm_handle = tqdm(ori_files)

    for ori_file in tqdm_handle:
        merge_pc(ori_file, div_files, args.resolution, args.div_dir, args.output_dir, args.task)

################################################################################
# Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='sample_dataset.py',
        description='Partition dataset into blocks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'ori_dir',
        help='Directory with original point cloud models.')
    parser.add_argument(
        'div_dir',
        help='Directory with point cloud blocks.')
    parser.add_argument(
        'output_dir',
        help='Directory where to save merged point clouds.')

    parser.add_argument(
        '--resolution',
        type=int, help='Resolution of blocks present in div_dir.', default=32)
    parser.add_argument(
        '--task', type=str, default='color',
        help='Compression tasks (geometry/color/geometry+color).')


    args = parser.parse_args()

    run(args)