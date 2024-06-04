import os
import sys
from annotator.dwpose import DWposeDetector
from pathlib import Path
import cv2

sys.path.append('.')

from utils import get_fps, read_frames, save_videos_from_pil
import numpy as np
from annotator.util import resize_image, HWC3
from PIL import Image
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--output_path", type=str)
    # parser.add_argument("--folder", type=str, help="Folder containing the video")
    parser.add_argument("--fps", type=str, help="Fps of video")
    parser.add_argument("--with_face", type=lambda x: (str(x).lower() == 'true'), help="Motion with face")
    args = parser.parse_args()

    video_path = args.video_path
    output_video_path = args.output_path
    with_face = args.with_face
    fps = int(args.fps)

    if not os.path.exists(video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    detector = DWposeDetector()

    orig_fps = get_fps(video_path)
    frames = read_frames(video_path)    

    kps_results = []
    for i, frame_pil in enumerate(frames):
        frame_pil = np.array(frame_pil, dtype=np.uint8)
        frame_pil = HWC3(frame_pil)
        frame_pil = resize_image(frame_pil, 512)
        result= detector(frame_pil, with_face)
        result = HWC3(result)
        img = resize_image(frame_pil, 512)
        H, W, C = img.shape
        result = cv2.resize(
            result, (W, H), interpolation=cv2.INTER_LINEAR
        )
        result = Image.fromarray(result)
        kps_results.append(result)

    print(output_video_path)
    save_videos_from_pil(kps_results, output_video_path, fps=orig_fps)

    if (int(orig_fps) != fps):
        clip = VideoFileClip(output_video_path)
        new_clip = clip.set_fps(fps)

    print(get_fps(output_video_path), output_video_path)