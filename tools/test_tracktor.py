import copy
import os
import time
from os import path as osp

import motmetrics as mm
import numpy as np
import sacred
import torch
import yaml
from sacred import Experiment
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracktor.datasets.factory import Datasets
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)
from torchreid.utils import FeatureExtractor

import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), 'src'))
from configs.path_cfg import MOTCHA_ROOT, OUTPUT_DIR

from mots.maskrcnn_fpn import MaskPredictor
from mots.mot2mots import generate_mots_file
from mots.evaluation import evaluate_mots

from tracktor.config import cfg as _cfg 

import os

if not osp.exists(_cfg.DATA_DIR):
    os.symlink(MOTCHA_ROOT, _cfg.DATA_DIR)


MOTS_SEQS  = [f'{seq:02}' for seq in (1, 2, 5, 6, 7, 9, 11, 12)]
mm.lap.default_solver = 'lap'


ex = Experiment()

ex.add_config('configs/tracktor.yaml')
#ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


# @ex.config
def add_reid_config(reid_models, obj_detect_models, dataset):
    # if isinstance(dataset, str):
    #     dataset = [dataset]
    if isinstance(reid_models, str):
        reid_models = [reid_models, ] * len(dataset)

    # if multiple reid models are provided each is applied
    # to a different dataset
    if len(reid_models) > 1:
        assert len(dataset) == len(reid_models)

    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)

    return reid_models, obj_detect_models, dataset


@ex.automain
def main(module_name, name, seed, obj_detect_models, reid_models, mots,
         tracker, oracle, dataset, load_results, frame_range, interpolate,
         write_images, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(OUTPUT_DIR, 'tracktor_logs', module_name, name)
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if mots['do_mots']:
        mots_output_dir = osp.join(OUTPUT_DIR, 'tracktor_logs', module_name + '_mots', name)
        os.makedirs(mots_output_dir, exist_ok=True)


    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(copy.deepcopy(_config), outfile, default_flow_style=False)

    dataset = Datasets(dataset)
    reid_models, obj_detect_models, dataset = add_reid_config(reid_models, obj_detect_models, dataset)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector(s).")
    obj_detects = []
    for obj_detect_model in obj_detect_models:
        obj_detect = FRCNN_FPN(num_classes=2)
        if not osp.exists(obj_detect_model):
            obj_detect_model = osp.join(OUTPUT_DIR, 'models', obj_detect_model)

        assert os.path.isfile(obj_detect_model)
        obj_detect_state_dict = torch.load(
            osp.join(OUTPUT_DIR, 'models', obj_detect_model), map_location=lambda storage, loc: storage)
        if 'model' in obj_detect_state_dict:
            obj_detect_state_dict = obj_detect_state_dict['model']

        obj_detect.load_state_dict(obj_detect_state_dict, strict=False)
        obj_detects.append(obj_detect)

        obj_detect.eval()
        if torch.cuda.is_available():
            obj_detect.cuda()

    # reid
    _log.info("Initializing reID network(s).")

    reid_networks = []
    for reid_model in reid_models:
        if not osp.exists(reid_model):
            reid_model = osp.join(OUTPUT_DIR, 'models', reid_model)

        assert os.path.isfile(reid_model)
        reid_network = FeatureExtractor(
            model_name='resnet50_fc512',
            model_path=reid_model,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        reid_networks.append(reid_network)

    # Segmentation:
    if mots['do_mots']:
        mask_model = MaskPredictor(num_classes=2)
        mask_model_path = mots['maskrcnn_model']
        if not osp.exists(mask_model_path):
            mask_model_path = osp.join(OUTPUT_DIR, 'models', mask_model_path)

        state_dict = torch.load(mask_model_path)['model']
        mask_model.load_state_dict(state_dict)
        mask_model.cuda()
        mask_model.eval()

    # tracktor
    if oracle is not None:
        tracker = OracleTracker(
            obj_detect, reid_network, tracker, oracle)
    else:
        tracker = Tracker(obj_detect, reid_network, tracker)

    time_total = 0
    num_frames = 0
    mot_accums = []
    eval_seqs = []

    for seq, obj_detect, reid_network in zip(dataset, obj_detects, reid_networks):
        # Messy way to evaluate on MOTS without having to modify code from the tracktor repo
        if mots['do_mots'] and mots['mots20_only'] and not any(str(seq).split('-')[1] in seq_ for seq_ in MOTS_SEQS):
            continue
        else:
            eval_seqs.append(str(seq))

        tracker.obj_detect = obj_detect
        tracker.reid_network = reid_network
        tracker.reset()

        _log.info(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)

        results = {}
        if load_results:
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()

            for frame_data in tqdm(seq_loader):
                with torch.no_grad():
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            _log.info(f"Tracks found: {len(results)}")
            _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            _log.info(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

            if mots['do_mots']:
                track_out_path = osp.join(output_dir, f"{str(seq)}.txt")
                seq_name = str(seq)
                if 'MOT17' in seq_name:
                    seq_name = seq_name[:8]
                    seq_name = seq_name.replace('MOT17', 'MOTS20')
                    
                mots_out_file = osp.join(mots_output_dir, f'{seq_name}.txt')
                seq_img_folder = osp.dirname(frame_data['img_path'][0])
                _log.info(f"Computing MOTS output for: {seq}")
                _log.info(f"Writing predictions to: {output_dir}")
                generate_mots_file(mask_model, track_out_path, seq_img_folder, mots_out_file)

        if seq.no_gt:
            _log.info("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq_loader))

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        _log.info("Evaluation:")
        evaluate_mot_accums(mot_accums,
                            [str(s) for s in dataset if not s.no_gt and str(s) in eval_seqs],
                            generate_overall=True)

    if mots['do_mots'] and mot_accums:
        evaluate_mots(mots_output_dir)