import os
import sys
sys.path.append('.')
from annotator.dwpose import DWposeDetector
from pathlib import Path
import cv2

from utils import read_frames, save_videos_from_pil
import numpy as np
from annotator.util import resize_image, HWC3
from PIL import Image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--video_folder", type=str, help="Folder containing the video")
    parser.add_argument("--fps", type=str, help="Specific fps")
    args = parser.parse_args()

    fps = args.fps
    video_path = os.path.join(args.folder, "video.mp4")
    output_path = args.output_path

    if not os.path.exists(video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    detector = DWposeDetector()
    frames = read_frames(video_path)    
    slice_frame = int(fps*4)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        frame_pil = np.array(frame_pil, dtype=np.uint8)
        frame_pil = HWC3(frame_pil)
        frame_pil = resize_image(frame_pil, 512)
        result= detector(frame_pil)
        result = HWC3(result)
        img = resize_image(frame_pil, 512)
        H, W, C = img.shape
        result = cv2.resize(
            result, (W, H), interpolation=cv2.INTER_LINEAR
        )
        result = Image.fromarray(result)
        kps_results.append(result)

    print(output_path)
    save_videos_from_pil(kps_results, output_path, fps=fps)
