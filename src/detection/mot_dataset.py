
import numpy as np
import os.path as osp

from vision.references.detection.coco_utils import ConvertCocoPolysToMask, CocoDetection, _coco_remove_images_without_annotations
from vision.references.detection.transforms import Compose


class UpdateIsCrowd(object):
    def __init__(self, min_size, min_vis = 0.2):
        self.min_size = min_size
        self.min_vis = min_vis

    def __call__(self, image, target):
        for i, ann in enumerate(target['annotations']):
            bbox = ann['bbox']
            bbox_too_small = max(bbox[-1],bbox[-2]) < self.min_size

            #print("HALLO checking vis")
            if 'vis' in ann:
                vis  = ann['vis']
            
            elif 'keypoints' in ann:
                vis = (np.array(ann['keypoints'])[2::3] ==2).mean().round(2)
            
            else:
                raise RuntimeError("The given annotations have no visibility measure. Are you sure you want to proceed?")
            
            not_vis = vis < self.min_vis
            target['annotations'][i]['iscrowd'] = max(ann['iscrowd'], int(bbox_too_small), int(not_vis))


        return image, target



def get_mot_dataset(img_folder, ann_file, transforms, min_size=25, min_vis = 0.2):
    t = [UpdateIsCrowd(min_size=min_size, min_vis=min_vis), ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    dataset = CocoDetection(img_folder=img_folder, 
                            ann_file=ann_file, 
                            transforms=transforms)  

    dataset = _coco_remove_images_without_annotations(dataset)

    return dataset