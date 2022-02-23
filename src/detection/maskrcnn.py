
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from torchvision.models.detection.mask_rcnn import MaskRCNN



def maskrcnn_resnet_fpn(backbone_name='resnet50',
    pretrained=False, num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs
):
    """Just modify the original torchvision function to accept arbitrary resnet backbones"""

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    backbone = resnet_fpn_backbone(backbone_name=backbone_name, pretrained=pretrained_backbone, norm_layer=misc_nn_ops.FrozenBatchNorm2d, trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    
    assert not pretrained

    return model
