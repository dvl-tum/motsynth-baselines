"""
TODOs: 
- argparse
- List sequences by number
- Get rid of asserts
"""

import pandas as pd

import os.path as osp
import os
import json

import configparser

import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', help="Directory path containing the 'annotations' directory with .json files")
    parser.add_argument('--save-path', help='Root file in which the new annoation files will be stored. If not provided, motsynth-root will be used')
    parser.add_argument('--save-dir', default='mots_annotations', help="name of directory within 'save-path'in which MOTS annotation files will be stored")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.motsynth_path

    return args

def save_seqinfo(seqinfo_path, info):
    seqinfo = configparser.ConfigParser()
    seqinfo.optionxform = str # Otherwise capital letters are ignored in keys

    seqinfo['Sequence'] = dict(name=info['seq_name'],
                            frameRate=info['fps'],
                            seqLength=info['sequence_length'],
                            imWidth= info['img_width'],
                            imHeight= info['img_height'],
                            weather=info['weather'],
                            time=info['time'],
                            isNight=info['is_night'],
                            isMoving=info['is_moving'],
                            FOV=info['cam_fov'],
                            imExt='.jpg',
                            fx=1158,
                            fy=1158,
                            cx=960,
                            cy=540)
        
    with open(seqinfo_path, 'w') as configfile:    # save
        seqinfo.write(configfile, space_around_delimiters=False)


def main(args):
    ann_dir = osp.join(args.motsynth_path, 'annotations')
    mots_ann_dir = osp.join(args.save_path, args.save_dir)
    
    seqs = [f'{seq_num:03}' for seq_num in range(768) if seq_num not in (629, 757, 524, 652)]

    for seq  in tqdm.tqdm(seqs):
        ann_path = osp.join(ann_dir, f'{seq}.json')
        with open(ann_path) as f:
            seq_ann = json.load(f)

        rows = []
        img_id2frame = {im['id']: im['frame_n'] for im in seq_ann['images']}    

        for ann in seq_ann['annotations']:
            assert ann['category_id'] == 1
            if ann['area']: # Include only objects with non-empty masks
                if not ann['iscrowd']:
                    mots_id = 2000 + ann['ped_id']
                
                else: # ID = 10000 means that the instance should be ignored during eval.
                    mots_id = 10000

                row = {'time_frame': img_id2frame[ann['image_id']],# STARTS AT 0!!!
                    'id': mots_id,
                    'class_id': 2,
                    'img_height': ann['segmentation']['size'][0],
                    'img_width': ann['segmentation']['size'][1],
                    'rle': ann['segmentation']['counts']}

                rows.append(row)
                
        # Save gt.txt file
        # Format in https://www.vision.rwth-aachen.de/page/mots        
        mots_ann = pd.DataFrame(rows, columns = ['time_frame', 'id', 'class_id', 'img_height', 'img_width', 'rle'])
        gt_dir = osp.join(mots_ann_dir, seq, 'gt')
        os.makedirs(gt_dir, exist_ok=True)
        mots_ann.to_csv(osp.join(gt_dir, 'gt.txt'), header=None, index=None, sep=' ')
        

        # Save seqinfo.ini
        seqinfo_path = osp.join(mots_ann_dir, seq, 'seqinfo.ini')
        save_seqinfo(seqinfo_path, info = seq_ann['info'])


if __name__ =='__main__':
    args = parse_args()
    main(args)