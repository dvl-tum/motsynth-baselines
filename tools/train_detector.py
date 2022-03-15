import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), 'src', 'detection'))
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), 'src', 'detection', 'torchvision_ref'))

from configs.path_cfg import MOTSYNTH_ROOT, MOTCHA_ROOT, OUTPUT_DIR

import datetime
import os
import time

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils

from engine import train_one_epoch, evaluate
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

from train import get_transform

try:
    from torchvision.prototype import models as PM
except ImportError:
    PM = None

from mot_dataset import get_mot_dataset
from maskrcnn import maskrcnn_resnet_fpn
from configs.path_cfg import OUTPUT_DIR

#get_motsyn_detection(root, image_set, transforms, min_size=25, min_vis = 0.25)
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--train-dataset", default="train", type=str, help="dataset name")
    parser.add_argument("--val-dataset", default="mot17", type=str, help="dataset name")

    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default='maskrcnn_training', type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=-1, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--backbone", default='resnet50', type=str, help="ResNet backbone used"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    # Prototype models only
    parser.add_argument(
        "--prototype",
        dest="prototype",
        help="Use prototype model builders instead those from main area",
        action="store_true",
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser

def get_dataset(name, data_path, image_set, transform,  img_path=None):
    if img_path is None:
        img_path = data_path

    if name == 'motsynth':
        ds = get_mot_dataset(img_path, osp.join(data_path, 'comb_annotations', f'{image_set}.json'), 
                transforms=transform)
    
    else:
        ds = get_mot_dataset(img_path, osp.join(data_path, 'motcha_coco_annotations', f'{name.upper()}_train.json'), 
                transforms=transform)

    return ds, 2

def main(args):
    if args.weights and PM is None:
        raise ImportError("The prototype module couldn't be found. Please install the latest torchvision nightly.")
    
    if args.output_dir:
        args.output_dir = osp.join(OUTPUT_DIR, 'detection_logs', args.output_dir)
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset('motsynth', MOTSYNTH_ROOT, args.train_dataset, get_transform(True, args))
    dataset_test, _ = get_dataset(args.val_dataset, MOTCHA_ROOT, None, get_transform(False, args))


    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")

    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    
    
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    if 'maskrcnn' in args.model:
        kwargs['backbone_name']= args.backbone
        model = maskrcnn_resnet_fpn(pretrained=args.pretrained, num_classes=num_classes, **kwargs)

    else:
        if not args.weights:
            model = torchvision.models.detection.__dict__[args.model](
                pretrained=args.pretrained, num_classes=num_classes, **kwargs
            )
        else:
            model = PM.detection.__dict__[args.model](weights=args.weights, num_classes=num_classes, **kwargs)
        

        
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        evaluate(model, data_loader_test, device=device, iou_types=['bbox'])
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device, iou_types=['bbox'])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
