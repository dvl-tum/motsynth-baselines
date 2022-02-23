import pandas as pd
import numpy as np

import os.path as osp
import os
import json

import tqdm
import argparse

from tools.anns.generate_mots_format_files import save_seqinfo

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', help="Directory path containing the 'annotations' directory with .json files")
    parser.add_argument('--save-path', help='Root file in which the new annoation files will be stored. If not provided, motsynth-root will be used')
    parser.add_argument('--save-dir', default='mot_annotations', help="name of directory within 'save-path'in which MOTS annotation files will be stored")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.motsynth_path

    return args

def main(args):
    ann_dir = osp.join(args.motsynth_path, 'annotations')
    mot_ann_dir = osp.join(args.save_path, args.save_dir)
    seqs = [f'{seq_num:03}' for seq_num in range(768) if seq_num not in (629, 757, 524, 652)]

    for seq  in tqdm.tqdm(seqs):
        ann_path = osp.join(ann_dir, f'{seq}.json')
        with open(ann_path) as f:
            seq_ann = json.load(f)

        rows = []
        img_id2frame = {im['id']: im['frame_n'] for im in seq_ann['images']}    

        for ann in seq_ann['annotations']:
            # We compute the 3D location as the mid point between both feet keypoints in 3D
            kps = np.array(ann['keypoints_3d']).reshape(-1, 4)
            feet_pos_3d = kps[[-1, -4], :3].mean(axis = 0).round(4)

            row = {'frame': img_id2frame[ann['image_id']],# STARTS AT 0!!!
                   'id': ann['ped_id'],
                   'bb_left': ann['bbox'][0] + 1, # Make it 1-based??
                   'bb_top': ann['bbox'][1] + 1,
                   'bb_width': ann['bbox'][2],
                   'bb_height': ann['bbox'][3],
                   'conf': 1 - ann['iscrowd'],
                   'class': 1 if ann['iscrowd'] == 0 else 8, # Class 8 means distractor. It is the one used by Trackeval as 'iscrowd' 
                   # We compute visibility as the proportion of visible keypoints
                   'vis': (np.array(ann['keypoints'])[2::3] ==2).mean().round(2),
                   'x': feet_pos_3d[0],
                   'y': feet_pos_3d[1],
                   'z': feet_pos_3d[2]}

            rows.append(row)
            
        # Save gt.txt file
        # Format in https://github.com/dendorferpatrick/MOTChallengeEvalKit/tree/master/MOT
        mot_ann = pd.DataFrame(rows, columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf','class', 'vis', 'x', 'y', 'z'])
        gt_dir = osp.join(mot_ann_dir, seq, 'gt')
        os.makedirs(gt_dir, exist_ok=True)
        mot_ann.to_csv(osp.join(gt_dir, 'gt.txt'), header=None, index=None, sep=',')
        

        # Save seqinfo.ini
        seqinfo_path = osp.join(mot_ann_dir, seq, 'seqinfo.ini')
        save_seqinfo(seqinfo_path, info = seq_ann['info'])

if __name__ =='__main__':
    args = parse_args()
    main(args)