# Installation
1. Create and activate an environment with all required packages:
```
conda env create -f environment.yml
conda activate motsynth
```
2. Install the additional modules: [`Tracktor`](https://github.com/phil-bergmann/tracking_wo_bnw), [`Torchreid`](https://github.com/KaiyangZhou/deep-person-reid/tree/master/torchreid) and [`Trackeval`](https://github.com/JonathonLuiten/TrackEval):
```
pip install https://github.com/phil-bergmann/tracking_wo_bnw/archive/master.zip
pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip
pip install https://github.com/JonathonLuiten/TrackEval/archive/master.zip
```
3.  Optionally modify the path `OUTPUT_DIR` in `configs/path_cfg.py`, download [TrackEval](https://github.com/JonathonLuiten/TrackEval)'s ground truth data from [here](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip) and place it in at `MOTCHA_ROOT`. E.g.:
```
MOTCHA_ROOT=$(python -c "from configs.path_cfg import MOTCHA_ROOT; print(MOTCHA_ROOT);")
wget -P $MOTCHA_ROOT https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip
```

4. Optionally modify the path `OUTPUT_DIR` in `configs/path_cfg.py`, and download our trained models:
```
OUTPUT_DIR=$(python -c "from configs.path_cfg import OUTPUT_DIR; print(OUTPUT_DIR);")
mkdir ${OUTPUT_DIR}/models
wget -P ${OUTPUT_DIR}/models https://vision.in.tum.de/webshare/u/brasoand/motsynth/resnet50_fc512_reid_epoch_19.pth
wget -P ${OUTPUT_DIR}/models https://vision.in.tum.de/webshare/u/brasoand/motsynth/maskrcnn_resnet50_fpn_epoch_10.pth
```