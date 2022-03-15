from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'configs'))
from path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT

from torchreid.data import ImageDataset
import tqdm

import pandas as pd
import numpy as np
import json

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def anns2df(anns, img_dir):
    if isinstance(anns['info'], list):
        assert len(anns['info']) == 1
        isnight = anns['info']['is_night']

    else: # Messy way to recover 'is_night' from combined annotation files
        imid2isnight = {im['id']: im['is_night'] for im in anns['images']}
        imid2frame = {im['id']: im['frame_n'] for im in anns['images']}
        isnight = {ann['id']: imid2isnight[ann['image_id']] for ann in anns['annotations']}

    rows = []
    for ann in tqdm.tqdm(anns['annotations']):        
        assert osp.exists(f"{img_dir}/{ann['id']}.png"), f"{img_dir}/{ann['id']}.png does not exist!!!"
        ##if not osp.exists(f"{img_dir}/{ann['id']}.png"): 
        #    continue

        row={'path': f"{img_dir}/{ann['id']}.png",
            'model_id': int(ann['model_id']),
            'height': int(ann['bbox'][-1]),
            'width': int(ann['bbox'][-2]),
            'iscrowd': int(ann['iscrowd']),
            'isnight': int(isnight if not isinstance(isnight, dict) else isnight[ann['id']]),
            'vis' : (np.array(ann['keypoints'])[2::3]).mean(),
            'frame_n': int(imid2frame[ann['image_id']]),
            **{f'attr_{i}': int(attr_val) for i, attr_val in enumerate(ann['attributes'])}}
        rows.append(row)

    return  pd.DataFrame(rows)

def anns2df_motcha(anns, img_dir):
    # Build DF from anns
    rows = []
    for ann in tqdm.tqdm(anns['annotations']):
        row={'path': f"{osp.join(img_dir)}/{ann['id']}.png",
            'ped_id': int(ann['ped_id']),
            'height': int(ann['bbox'][-1]),
            'width': int(ann['bbox'][-2]),
            'iscrowd': int(ann['iscrowd']),
            'vis' : float(ann['vis']),
            'frame_n': int(ann['frame_n'])}
        rows.append(row)

    return  pd.DataFrame(rows)

def assign_ids(df, night_id=True, attr_indices = [0, 2, 3, 4, 7, 9, 10]): #attr_indices = [0, 2, 3, 4, 7, 8, 9, 10]):
    id_cols = ['model_id'] + [f'attr_{i}' for i in attr_indices if f'attr{i}' in df.columns] 
    if night_id and 'isnight' in df.columns:
        id_cols += ['isnight']

    unique_ids_df = df[id_cols].drop_duplicates()
    unique_ids_df['reid_id'] = np.arange(unique_ids_df.shape[0])

    return  df.merge(unique_ids_df)

def clean_rows(df, min_vis, min_h, min_w, min_samples):
    # Filter by size and occlusion
    keep = (df['vis'] >= min_vis) & (df['height']>=min_h) & (df['width'] >= min_w) & (df['iscrowd']==0)
    clean_df = df[keep]

    # Keep only ids with at least MIN_SAMPLES appearances
    clean_df['samples_per_id'] = clean_df.groupby('reid_id')['height'].transform('count').values
    clean_df = clean_df[clean_df['samples_per_id']>=min_samples]

    return clean_df

def relabel_ids(df):
    df.rename(columns = {'reid_id': 'reid_id_old'}, inplace=True)

    # Relabel Ids from 0 to N-1
    ids_df = df[['reid_id_old']].drop_duplicates()
    ids_df['reid_id'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)
    return df


class MOTSeqDataset(ImageDataset):
    def __init__(self,ann_file, img_dir, min_vis=0.25, min_h=50, min_w=25, min_samples=15, night_id=True, motcha=False, **kwargs):        

        print("Reading json...")
        anns = read_json(ann_file)
        print("Done!")
        
        print("Preparing dataset...")
        if motcha:
            df = anns2df_motcha(anns, img_dir)
            df['reid_id'] = df['ped_id']

        else:
            df = anns2df(anns, img_dir)
            df = assign_ids(df, night_id=night_id, attr_indices = [0, 2, 3, 4, 7, 8, 9, 10])

        df= clean_rows(df, min_vis, min_h=min_h, min_w=min_w, min_samples=min_samples)
        df = relabel_ids(df)

        # For testing, choose one apperance randomly for every track and put in the gallery
        to_tuple_list = lambda df: list(df[['path', 'reid_id', 'cam_id']].itertuples(index=False, name=None))
        df['cam_id'] = 0
        train = to_tuple_list(df)

        df['index'] = df.index.values
        np.random.seed(0)
        query_per_id = df.groupby('reid_id')['index'].agg(lambda x: np.random.choice(list(x.unique())))
        query_df = df.loc[query_per_id.values].copy()
        gallery_df = df.drop(query_per_id).copy()
        gallery_df['cam_id'] = 1

        query=to_tuple_list(query_df)
        gallery=to_tuple_list(gallery_df)

        print("Done!")
        super(MOTSeqDataset, self).__init__(train, query, gallery, **kwargs)
        

def get_sequence_class(seq_name):
    """Hacky way to reuse the same class code for different sequences. We wrap the same class and 
    change the __name__ attribute.
    """

    if 'MOT17' in seq_name:
        ann_file = osp.join(MOTCHA_ROOT, f'motcha_coco_annotations/{seq_name}.json')
        img_dir = osp.join(MOTCHA_ROOT, 'reid_images')
        min_samples = 5
        motcha=True

    else:
        #assert seq_name in [f'motsynth_{split}{maybe_mini}' for split in [1, 2, 3, 4, 'train', 'val'] for maybe_mini in ('_mini', '')]
        split_name = seq_name.split('motsynth_')[-1]
        ann_file = osp.join(MOTSYNTH_ROOT, 'comb_annotations', f'{split_name}.json')
        img_dir = osp.join(MOTSYNTH_ROOT, 'reid')
        min_samples = 15
        motcha=False

    class MOTSeq(MOTSeqDataset):
        def __init__(self, **kwargs):
            super(MOTSeq, self).__init__(ann_file = ann_file, img_dir=img_dir, min_samples=min_samples, motcha=motcha,**kwargs)

    MOTSeq.__name__=seq_name
    
    return MOTSeq
        
        
        
        
