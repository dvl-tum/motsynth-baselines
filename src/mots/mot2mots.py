import numpy as np

import torch
from matplotlib.pyplot import imread
import os.path as osp
from torchvision.transforms import ToTensor
import pandas as pd
from pycocotools import mask
import tqdm

import csv

class MOT2MOTSDataset(torch.utils.data.Dataset):
    def __init__(self, det_path, seq_img_folder):
        self.det_df = pd.read_csv(det_path, sep=",", header=None)
        self.det_df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
        self.seq_img_folder = seq_img_folder
        self.transform = ToTensor()

    def __len__(self):
        return len(self.det_df['frame'].unique())

    def __getitem__(self, ix):
        f = self.det_df['frame'].unique()[ix]
        frame_detects = self.det_df.loc[self.det_df['frame'] == f]

        # Frame number
        frame_nums = frame_detects['frame'].to_numpy()
        frame_nums = torch.from_numpy(frame_nums)

        # Get the image
        img_prefix = '{:06d}'.format(int(f)) + '.jpg'
        img_path = osp.join(self.seq_img_folder, img_prefix)
        img = imread(img_path)
        img = self.transform(img)

        # Detection id
        ids = frame_detects['id'].to_numpy()
        ids = torch.from_numpy(ids)

        # Get bboxes
        bboxes = frame_detects[['bb_left', 'bb_top', 'bb_width', 'bb_height']].to_numpy()
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        bboxes -= 1 # Files are 1-based

        bboxes = torch.from_numpy(bboxes).float()

        return frame_nums, img, ids, bboxes

def ensure_unique_masks(frame_masks):
    """
    Ensures that there are not overlapping pixels in the masks of a frame. This is a MOTS challenge submission
    requirement

    frame_masks: np.array (N, H, W)
    returns: np.array (N, H, W)
    """
    safe_masks = np.zeros_like(frame_masks)

    # Get the maximum mask ix
    i = np.argmax(frame_masks, axis=0)

    # Create indices to access max elements
    h, w = np.indices((frame_masks.shape[1], frame_masks.shape[2]))
    ix = (i, h, w)

    # Copy max elements
    safe_masks[ix] = frame_masks[ix]

    return safe_masks


def generate_mots_file(mask_predictor, track_out_path, seq_img_folder, out_file):    
    mot2mots_dataset = MOT2MOTSDataset(track_out_path, seq_img_folder)
    mot2mots_dataloader = torch.utils.data.DataLoader(mot2mots_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                                      num_workers=0)

    for frame_num, frame_img, ids, roi_boxes in tqdm.tqdm(mot2mots_dataloader):
        frame_num = frame_num.squeeze(0).numpy()
        ids = ids.squeeze(0).numpy()
        roi_boxes = roi_boxes.squeeze(0)

        with torch.no_grad():
            mask_predictor.load_image(frame_img)
            pred_masks = mask_predictor.predict_masks(roi_boxes).squeeze(1).cpu().numpy()

        pred_masks = ensure_unique_masks(pred_masks)
        pred_masks = np.where(pred_masks >= 0.5, 1, 0).astype(np.uint8)

        with open(out_file, "a") as pred_f:
            writer = csv.writer(pred_f, delimiter=' ')
            for ix in range(ids.shape[0]):
                rle = mask.encode(np.asarray(pred_masks[ix].astype("uint8"), order="F"))
                # [Frame Id Class im_height im_width rle]
                writer.writerow([frame_num[ix], int(2000 + ids[ix]), 2, rle['size'][0], rle['size'][1],
                                 rle['counts'].decode(encoding='UTF-8')])