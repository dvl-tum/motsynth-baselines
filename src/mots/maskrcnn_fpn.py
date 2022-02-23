from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.roi_heads import maskrcnn_inference
from torchvision.models.detection.roi_heads import paste_masks_in_image
from tracktor.frcnn_fpn import FRCNN_FPN

# Modified from FRCNN_FPN
class MaskPredictor(FRCNN_FPN, MaskRCNN):
    def __init__(self, num_classes):
        backbone = resnet_fpn_backbone('resnet50', False)
        MaskRCNN.__init__(self, backbone, num_classes)
        #MaskRCNN.__init__(self, backbone, num_classes)

        # Cache for feature use
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None

    def predict_masks(self, boxes, return_roi_masks=False):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]

        # Get masks
        labels = [torch.ones((len(proposals[0]),), dtype=torch.int64)] # Set person label for coco
        mask_features = self.roi_heads.mask_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        mask_features = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_features)
        mask_probs = maskrcnn_inference(mask_logits, labels)

        if return_roi_masks:
            return mask_probs[0]

        boxes = resize_boxes(boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])
        pred_masks = paste_masks_in_image(mask_probs[0], boxes, self.original_image_sizes[0])
        return pred_masks