import argparse
import os
import os.path as osp
import cv2
import glob

import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
from configs.path_cfg import MOTSYNTH_ROOT
import tqdm


def main():
    parser = argparse.ArgumentParser(description='Get frames from a video')
    parser.add_argument('--motsynth-root', help='Directory hosting MOTSYnth part directories', default=MOTSYNTH_ROOT)
    args = parser.parse_args()

    video_paths = glob.glob(osp.join(args.motsynth_root, 'MOTSynth_[0-9]/[0-9][0-9][0-9].mp4'))

    frames_dir = os.path.join(args.motsynth_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("Start extracting frames...")

    for video_path in tqdm.tqdm(video_paths):
        vidcap = cv2.VideoCapture(video_path)

        seq_name = osp.basename(video_path).split(".")[0].zfill(3)
        out_dir = os.path.join(frames_dir, seq_name, 'rgb')
        os.makedirs(out_dir, exist_ok=True)

        count = 1
        success = True

        #print("Unpacking video...")

        while success:
            success, image = vidcap.read()
            if count < 3:
                count += 1
                continue
            if not success or count == 1803:
                break
            if count%200 == 0:
                print("Extract frames until: " + str(count - 3).zfill(4) + ".jpg")
            filename = os.path.join(out_dir, str(count - 3).zfill(4) + ".jpg")
            cv2.imwrite(filename, image)     # save frame as JPEG file
            count += 1

    print("Done!")

if __name__ == '__main__':
    main()