import os
import os.path as osp
import json
import argparse
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motsynth-path', help="Directory path containing the 'annotations' directory with .json files")
    parser.add_argument('--save-path', help='Root file in which the new annoation files will be stored. If not provided, motsynth-root will be used')
    parser.add_argument('--save-dir', default='comb_annotations', help="name of directory within 'save-path'in which MOTS annotation files will be stored")
    parser.add_argument('--subsample', default=10, type=int, help="Frame subsampling rate. If e.g. 10 is selected, then we will select 1 in 10 frames")
    parser.add_argument('--split', default='train', help="Name of split (i.e. set of sequences being merged) being used. A file named '{args.split}.txt needs to exist in the splits dir")
    parser.add_argument('--name', help="Name of the split that file that will be generated. If not provided, the split name will be used")
    
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.motsynth_path
    
    if args.name is None:
        args.name = args.split

    assert args.subsample >0, "Argument '--subsample' needs to be a positive integer. Set it to 1 to use every frame"

    return args
    
def read_split_file(path):
    with open(path, 'r') as file:
        seq_list = file.read().splitlines()

    return seq_list

def main(args):    
    # Determine which sequences to use
    seqs = [seq.zfill(3) for seq in read_split_file(osp.join(osp.dirname(os.path.abspath(__file__)), 'splits', f'{args.split}.txt'))]
    comb_anns  = {'images': [], 'annotations': [], 'categories': None, 'info': {}}

    for seq in tqdm.tqdm(seqs):
        ann_path = osp.join(args.motsynth_path, 'annotations',  f'{seq}.json')
        with open(ann_path) as f:
            seq_ann = json.load(f)

        # Subsample images and annotations if needed
        if args.subsample >1:
            seq_ann['images'] = [{**img, **seq_ann['info']} for img in seq_ann['images'] if ((img['frame_n'] -1 )% args.subsample) == 0] # -1 bc in the paper this was 0-based 
            img_ids = [img['id'] for img in seq_ann['images']]
            seq_ann['annotations'] = [ann for ann in seq_ann['annotations'] if ann['image_id'] in img_ids]

        comb_anns['images'].extend(seq_ann['images'])
        comb_anns['annotations'].extend(seq_ann['annotations'])
        comb_anns['info'][seq] = seq_ann['info']
    
    if len(seqs) > 0:
        comb_anns['categories'] = seq_ann['categories']
        comb_anns['licenses'] = seq_ann['categories']

    # Sanity check:
    img_ids = [img['id'] for img in comb_anns['images']]
    ann_ids = [ann['id'] for ann in comb_anns['annotations']]
    assert len(img_ids) == len(set(img_ids))
    assert len(ann_ids) == len(set(ann_ids))

    # Save the new annotations file
    comb_anns_dir = osp.join(args.save_path, args.save_dir)
    os.makedirs(comb_anns_dir, exist_ok=True)
    comb_anns_path = osp.join(comb_anns_dir, f"{args.name}.json")
    with open(comb_anns_path, 'w') as json_file:
        json.dump(comb_anns, json_file)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)