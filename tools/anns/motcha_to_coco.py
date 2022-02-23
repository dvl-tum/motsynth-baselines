import os
import os.path as osp
import numpy as np
import json

import argparse
import configparser
import datetime

import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', help="Path containing the dataset in a folder")
    parser.add_argument('--dataset', default='MOT17', help='Name of the dataset to be used. Should be either MOT17 or MOT20')
    parser.add_argument('--save-dir', help='Root file in which the new annoation files will be stored. If not provided, data-root will be used')
    parser.add_argument('--split', default='train', help="Split processed within the dataset. Should be either 'train' or 'test'")
    parser.add_argument('--save-combined', default=True, action='store_true', help="Determines whether a separate .json file containing all sequence annotations will be created")    
    parser.add_argument('--subsample', default=1, type=int, help="Frame subsampling rate. If e.g. 10 is selected, then we will select 1 in 10 frames")

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = osp.join(args.data_root, 'motcha_coco_annotations')

    return args

def get_img_id(dataset, seq, fname):
    # Dataset num, seq num, frame num
    return int(f"{dataset[3:5]}{seq.split('-')[1]}{int(fname.split('.')[0]):06}")

def read_seqinfo(path):
    cp = configparser.ConfigParser()
    cp.read(path)
    return {'height': int(cp.get('Sequence', 'imHeight')),
            'width': int(cp.get('Sequence', 'imWidth')),
            'fps': int(cp.get('Sequence', 'frameRate'))}
    
def main(args):
    data_path = osp.join(args.data_root, args.dataset, args.split)
    seqs = os.listdir(data_path)
    
    if args.save_combined:
        comb_data = {'info': {'dataset': args.dataset,
                              'split': args.split,
                              'creation_date': datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')},
                'images': [], 
                'annotations': [], 
                'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}

    for seq in tqdm.tqdm(seqs):
        if args.dataset.lower() == 'mot17':
            if not seq.endswith('DPM'): # Choose an arbitrary set of detections for MOT17, annotations are the same
                continue
                

        print(f"Processing sequence {seq} in dataset {args.dataset}")

        seq_path = osp.join(data_path, seq)
        seqinfo_path = osp.join(seq_path, 'seqinfo.ini')
        gt_path = osp.join(seq_path, 'gt/gt.txt')
        im_dir = osp.join(seq_path, 'img1')

        if args.dataset.lower() == 'mot17':
            seq_ = '-'.join(seq.split('-')[:-1]) # Get rid of detector string
        
        else:
            seq_ = seq.copy()

        
        seqinfo = read_seqinfo(seqinfo_path)
        data = {'info': {'sequence': seq_,
                         'dataset': args.dataset,
                         'split': args.split,
                         'creation_date': datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'),
                          **seqinfo},
                'images': [], 
                'annotations': [], 
                'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}

        # Load Bounding Box annotations
        gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=',')
        keep_classes = [1, 2, 7, 8, 12]
        mask = np.isin(gt[:, 7], keep_classes)
        gt = gt[mask]
        anns = [{'ped_id': row[1],
                'frame_n': row[0],     
                'category_id': 1,
                'id': f"{get_img_id(args.dataset, seq, f'{int(row[0]):06}.jpg')}{int(row_i):010}",
                'image_id': get_img_id(args.dataset, seq, f'{int(row[0]):06}.jpg'),
                #'bbox': row[2:6].tolist(),
                'bbox': [row[2] - 1, row[3] - 1, row[4], row[5]], # MOTCha annotations are 1-based
                'area': row[4]*row[5],
                'vis': row[8],
                'iscrowd': 1 - row[6]}
            for row_i, row in enumerate(gt.astype(float)) if row[0]% args.subsample ==0]

        # Load Image information 
        all_img_ids  =list(set([aa['image_id'] for aa in anns]))         
        imgs = [{'file_name': osp.join(args.dataset, args.split, seq, 'img1', fname),
                 'height': seqinfo['height'], 
                 'width': seqinfo['width'],
                 'frame_n': int(fname.split('.')[0]),
                 'id': get_img_id(args.dataset, seq, fname)}                    
                for fname in os.listdir(im_dir) if get_img_id(args.dataset, seq, fname) in all_img_ids]
        assert len(set([im['id'] for im in imgs])) == len(imgs)
        data['images'].extend(imgs)
        assert len(str(imgs[0]['id'])) == len(str(anns[0]['image_id']))
        
        data['annotations'].extend(anns)
        
        os.makedirs(args.save_dir, exist_ok=True)
        fname = f"{args.dataset}_{seq_}.json" if args.dataset not in seq_ else f"{seq_}.json"
        save_path = osp.join(args.save_dir, fname)
        with open(save_path, 'w') as f:
            json.dump(data, f)

        print(f"Saved result at {save_path}")

        if args.save_combined:
            comb_data['annotations'].extend(anns)
            comb_data['images'].extend(imgs)

    if args.save_combined:
        save_path = osp.join(args.save_dir, f"{args.dataset}_{args.split}.json")
        with open(save_path, 'w') as f:
            json.dump(comb_data, f)

        print(f"Saved combined result at {save_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

        
        

