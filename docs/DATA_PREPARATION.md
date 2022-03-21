
# Data preparation
## Setup
- You can optionally modify `MOTCHA_PATH` and `MOTSYNTH_PATH` and `OUTPUT_DIR` as your directories for you MOT17, MOTSynth, and you train/eval outputs at `configs/path_cfg.py`.

## Downloading and preparing MOTSynth

1. Download and extract all MOTSynth videos. This will take a while...
```
MOTSYNTH_ROOT=$(python -c "from configs.path_cfg import MOTSYNTH_ROOT; print(MOTSYNTH_ROOT);")
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_1.zip
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_2.zip
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_3.zip

unzip $MOTSYNTH_ROOT/MOTSynth_1.zip -d $MOTSYNTH_ROOT
unzip $MOTSYNTH_ROOT/MOTSynth_2.zip -d $MOTSYNTH_ROOT
unzip $MOTSYNTH_ROOT/MOTSynth_3.zip -d $MOTSYNTH_ROOT

rm $MOTSYNTH_ROOT/MOTSynth_1.zip
rm $MOTSYNTH_ROOT/MOTSynth_2.zip
rm $MOTSYNTH_ROOT/MOTSynth_3.zip
```
2. Extract frames from the videos you downloaded. Again, this will take while.
```
python tools/anns/to_frames.py --motsynth-root $MOTSYNTH_ROOT

# You can now delete the videos
rm -r $MOTSYNTH_ROOT/MOTSynth_1
rm -r $MOTSYNTH_ROOT/MOTSynth_2
rm -r $MOTSYNTH_ROOT/MOTSynth_3
```
3. Download and extract the annotations (in several formats):
```
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_coco_annotations.zip
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_mot_annotations.zip
wget -P $MOTSYNTH_ROOT https://motchallenge.net/data/MOTSynth_mots_annotations.zip
# Merged annotation files for ReID and detection trainings
wget -P $MOTSYNTH_ROOT https://vision.in.tum.de/webshare/u/brasoand/motsynth/comb_annotations.zip

unzip $MOTSYNTH_ROOT/MOTSynth_coco_annotations.zip -d $MOTSYNTH_ROOT
unzip $MOTSYNTH_ROOT/MOTSynth_mot_annotations.zip -d $MOTSYNTH_ROOT
unzip $MOTSYNTH_ROOT/MOTSynth_mots_annotations.zip -d $MOTSYNTH_ROOT
unzip $MOTSYNTH_ROOT/comb_annotations.zip -d $MOTSYNTH_ROOT

rm $MOTSYNTH_ROOT/MOTSynth_coco_annotations.zip
rm $MOTSYNTH_ROOT/MOTSynth_mot_annotations.zip
rm $MOTSYNTH_ROOT/MOTSynth_mots_annotations.zip
rm $MOTSYNTH_ROOT/comb_annotations.zip
```
**Note**: You can generate the mot, mots and combined annotation files yourself from the original coco format annotations with the scripts `tools/anns/generate_mot_format_files.py`, `tools/anns/generate_mots_format_files.py`, and `tools/anns/combine_anns.py`, respectively.

After runnning these steps, your `MOTSYNTH_ROOT` directory should look like this:
```text
$MOTSYNTH_ROOT
├── frames
    │-- 000
    │   │-- rgb
    │   │   │-- 0000.jpg
    │   │   │-- 0001.jpg
    │   │   │-- ...
    │-- ...
├── annotations
    │-- 000.json
    │-- 001.json
    │-- ...
├── comb_annotations
    │-- split_1.json 
    │-- split_2.json
    │-- ...
├── mot_annotations
    │-- 000
    │   │-- gt
    │   │   │-- gt.txt
    │   │-- seqinfo.ini
    │-- ...
├── mots_annotations
    │-- 000
    │   │-- gt
    │   │   │-- gt.txt
    │   │-- seqinfo.ini
    │-- ...

```


## Downloading and preparing MOT17
We will use MOT17 for both tracking and MOTS experiments, since MOTS20 sequences are a subset of MOT17 sequences. To download it, follow these steps:

1. Download and extract it under `$MOTCHA_ROOT`. E.g.:
```
MOTCHA_ROOT=$(python -c "from configs.path_cfg import MOTCHA_ROOT; print(MOTCHA_ROOT);")
wget -P $MOTCHA_ROOT https://motchallenge.net/data/MOT17.zip
unzip $MOTCHA_ROOT/MOT17.zip -d $MOTCHA_ROOT
rm $MOTCHA_ROOT/MOT17.zip
```
2. Download and extract COCO-format MOT17 annotations (or alternatively, you can generate them with `tools/anns/motcha_to_coco.py`). These are needed for evaluation in detection and reid trainings.
```
wget -P $MOTCHA_ROOT https://vision.in.tum.de/webshare/u/brasoand/motsynth/motcha_coco_annotations.zip
unzip $MOTCHA_ROOT/motcha_coco_annotations.zip -d $MOTCHA_ROOT
rm $MOTCHA_ROOT/motcha_coco_annotations.zip
```

After runnning these steps, your `MOTCHA_ROOT` directory should look like this:
```
$MOTCHA_ROOT
├── MOT17
|   │-- train
|   │   │-- MOT17-02-DPM
|   │   │   │-- gt
|   │   │   │   |-- gt.txt      
|   │   │   │-- det
|   │   │   │   |-- det.txt    
|   │   │   |-- img1
|   │   │   │   |-- 000001.jpg
|   │   │   │   |-- 000002.jpg
|   │   │   │   |-- ...
|   │   │   │-- seqinfo.ini    
|   |   |-- MOT17-02-FRCNN
|   │   │   │-- ...    
|   |   |-- ...
|   │-- test
|       │-- MOT17-01-DPM 
|       │-- ...
|       
|--motcha_coco_annotations
   │-- MOT17-02.json 
   │-- ...
   │-- MOT17-train.json 
```
## ReID data
**Note**: This is only needed if you want to train you own ReID model.

To train and evaluate ReID models, we store the bounding-box cropped images of pedestrians in every 60th frame from both MOTSynth and MOT17, respectively. You can download these images here:
```
# For MOT17
MOTCHA_ROOT=$(python -c "from configs.path_cfg import MOTCHA_ROOT; print(MOTCHA_ROOT);")
wget -P $MOTCHA_ROOT https://vision.in.tum.de/webshare/u/brasoand/motsynth/motcha_reid_images.zip.zip
unzip $MOTCHA_ROOT/motcha_reid_images.zip -d $MOTCHA_ROOT
rm $MOTCHA_ROOT/motcha_reid_images.zip

# For MOTSynth
MOTSYNTH_ROOT=$(python -c "from configs.path_cfg import MOTSYNTH_ROOT; print(MOTSYNTH_ROOT);")
wget -P $MOTSYNTH_ROOT https://vision.in.tum.de/webshare/u/brasoand/motsynth/motsynth_reid_images.zip.zip
unzip $MOTSYNTH_ROOT/motsynth_reid_images.zip -d $MOTSYNTH_ROOT
rm $MOTSYNTH_ROOT/motsynth_reid_images.zip

```

Alternatively, you can directly generate these images locally by running:
```
# For MOT17
python tools/anns/store_reid_imgs.py --ann-path $MOTCHA_ROOT/motcha_coco_annotations/MOT17_train.json

# For MOTSynth
python tools/anns/store_reid_imgs.py --ann-path $MOTSYNTH_ROOT/comb_annotations/train_mini.json
```
