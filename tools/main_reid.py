import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), 'src', 'reid'))
#sys.path.append(osp.dirname(osp.dirname(__file__)))
from configs.path_cfg import OUTPUT_DIR
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import time
import os.path as osp
import argparse
import torch
import torch.nn as nn
import os

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

from torchreid.data.datasets import __image_datasets
from mot_reid_dataset import get_sequence_class
from engine import ImageSoftmaxEngineSeveralSeq

def update_datasplits(cfg):
    def get_split_seqs(name):
        splits = ['split_1', 'split_2', 'split_3', 'split_4', 'train', 'val']
        motsynth_splits= [f'motsynth_{split}{maybe_mini}' for split in splits for maybe_mini in ('_mini', '')]
        assert name in ['market1501', 'cuhk03', 'mot17']+motsynth_splits, f"Got dataset name {name}"
        if name in ('market1501', 'cuhk03', 'motsynth_train'):
            return name
        
        if name == 'mot17':
            return [f'MOT17-{seq_num:02}' for seq_num in (2, 4, 5, 9, 10, 11, 13)]

        return name
            
    assert isinstance(cfg.data.sources, (tuple, list))
    assert isinstance(cfg.data.sources, (tuple, list))
    cfg.data.sources = [get_split_seqs(ds_name) for ds_name in cfg.data.sources]
    cfg.data.targets = [get_split_seqs(ds_name) for ds_name in cfg.data.targets]
    
    if isinstance(cfg.data.sources[0], (tuple, list)) and len(cfg.data.sources) == 1:
        cfg.data.sources = cfg.data.sources[0]
    
    if isinstance(cfg.data.targets[0], (tuple, list)) and len(cfg.data.targets) == 1:
        cfg.data.targets = cfg.data.targets[0]

def register_datasets(cfg):
    for maybe_data_list in (cfg.data.sources, cfg.data.targets):
        if not isinstance(maybe_data_list, (tuple, list)):
            maybe_data_list = [maybe_data_list]
            
        for seq_name in maybe_data_list:
            #print("Registering dataset ", seq_name)
            if seq_name not in __image_datasets:
                seq_class = get_sequence_class(seq_name)
                torchreid.data.register_image_dataset(seq_name, seq_class)

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))

def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = ImageSoftmaxEngineSeveralSeq(
            #engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine

def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms

def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(OUTPUT_DIR, 'reid_logs', cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    update_datasplits(cfg)        
    register_datasets(cfg)

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))

if __name__ == '__main__':
    main()
