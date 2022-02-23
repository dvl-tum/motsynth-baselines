import os
import os.path as osp
import numpy as np
import json

import argparse

import tqdm
from PIL import Image
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-path', help=".JSON annotations file path in COCO format")
    parser.add_argument('--frames-path', help="Root directory containing images")
    parser.add_argument('--save-dir', help='Root file in which the new annoation files will be stored. If not provided, data-root will be used')
    parser.add_argument('--start-iter', default=0, type=int)
    
    #args = parser.parse_args(['--ann-path', '/storage/user/brasoand/MOTSynth/comb_annotations/train_mini.json'])
    args = parser.parse_args()

    if args.frames_path is None:
        args.frames_path = osp.dirname(osp.dirname(args.ann_path))

    if args.save_dir is None:
        #args.save_dir = osp.join(osp.dirname(osp.dirname(args.ann_path)), 'reid_images')
        args.save_dir = osp.join(osp.dirname(osp.dirname(args.ann_path)), 'reid')


    return args

def crop_box(im, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+ w, y1+ h
    return im.crop((x1, y1, x2, y2))


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # Read annotations
    with open(args.ann_path) as f:
        anns = json.load(f)


    # Annotation ids are used as file names to store boxes. 
    # Therefore they need to be unique
    ann_ids = [ann['id'] for ann in anns['annotations']]
    assert len(ann_ids) == len(set(ann_ids))    
    imgid2file = {img['id']: img['file_name'] for img in anns['images']}

    # TODO: This needs to go!!!!!!!    
    anns['annotations'] = [ann for ann in anns['annotations'] if not osp.exists(osp.join(args.save_dir, f"{ann['id']}.png"))]
    len(anns['annotations'])
    im2anns = defaultdict(list)
    for ann in anns['annotations']:
        im2anns[imgid2file[ann['image_id']]].append(ann)

    for img_file, im_anns  in tqdm.tqdm(im2anns.items()):
        #break
        # Read Image
        im_path = osp.join(args.frames_path, img_file)
        if not osp.exists(im_path):
            im_path = osp.join(args.frames_path, img_file.replace('rgb/', ''))

        assert osp.exists(im_path)
        im = Image.open(im_path)

        for ann in im_anns:
            box_path = osp.join(args.save_dir, f"{ann['id']}.png")
            
            if osp.exists(box_path):
                continue

            #if ann['bbox'][-2] > 2000 or ann['bbox'][-1] > 2000:
            #    continue

            box_im = crop_box(im, ann['bbox'])
            box_im.save(box_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
