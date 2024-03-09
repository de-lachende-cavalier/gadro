"""
This Python file contains a utility for extracting single frames from a raw video, in PNG format.

I have provided my own version because the one that came with the dataset didn't work...
"""

import cv2
import os


def extract_frames_from_video(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_file = f"frame_{frame_idx}.png"
        frame_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(frame_path, frame)

        frame_idx += 1
    video.release()


video_path = "find_dataset/Our_database/raw_videos/"

for name in os.listdir(video_path):
    if name.endswith("mp4"):  # mp4 files only
        output_folder = f"find_dataset/frames/{name[:3]}"
        extract_frames_from_video(video_path + name, output_folder)
        print(f"[+] Done extracting frames for {name}.")
