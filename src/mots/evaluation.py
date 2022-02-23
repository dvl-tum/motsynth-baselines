import argparse
from multiprocessing import freeze_support

import sys
import os.path as osp
sys.path.append(osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), 'configs'))
from path_cfg import MOTCHA_ROOT

import trackeval  # noqa: E402
import os.path as osp



#evaluate_mots('/storage/slurm/brasoand/motsynth_output/mots_private_det_095_mots/', gt_dir=None)
def evaluate_mots(output_dir, gt_dir=None, dataset_name= 'MOTS-train'):
    """
    Source: https://github.com/JonathonLuiten/TrackEval/blob/master/scripts/run_mots_challenge.py
    Only modified args.
    """
    if gt_dir is None:
        gt_dir = MOTCHA_ROOT

    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MOTSChallenge.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args([]).__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Update config entries!
    assert osp.isdir(output_dir) 
    dataset_config['TRACKERS_FOLDER'] = osp.dirname(output_dir)
    dataset_config['SEQMAP_FILE'] = osp.join(gt_dir, f'data/gt/mot_challenge/seqmaps/{dataset_name}.txt')
    dataset_config['GT_FOLDER'] = osp.join(gt_dir, f'data/gt/mot_challenge/{dataset_name}')


    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKER_SUB_FOLDER'] = ''

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE,
                   trackeval.metrics.JAndF]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)