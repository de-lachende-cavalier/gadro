"""
This Python file contains utilities for directly interacting with the data in find_dataset/.

The functions whose name begins with an underscore are considered "private", i.e., only meant for internal use within this file. They're undocumented because they're quite self explanatory and short.

Functions:
    get_mat_data - Returns the MATLAB data in a format suitable for python (numpy arrays), together with some extra information (the number of subjects and the number of frames), given a certain .mat filename.

    get_frames_by_type - Returns either one frame or all the frames pertaining to a video, given their type (whether they come from find_dataset/frames or find_dataset/dynamic).

    get_feature_frames - Combines speaker and non-speaker marged maps for each frame into a single frame (instead of two) and generates a frame highlighting the speaker (i.e., a bitmap where the white patch designates the speaker).
"""

import os
import re
import cv2
import numpy as np
from scipy.io import loadmat


def get_mat_data(mat_filename):
    base = "find_dataset/Our_database/fix_data_NEW/"

    mat_data = loadmat(base + mat_filename)

    # (39, 1) ndarray, one entry per subject, to get the actual data do
    fix_data = mat_data["curr_v_all_s"]  # (39, 1) ndarray
    num_subjects = fix_data.shape[0]
    # fix_data[subject_idx][0] => (num_frames, 2) ndarray
    num_frames = fix_data[0][0].shape[0]  # every subject watches the same video

    return fix_data, num_subjects, num_frames


def get_frames_by_type(frame_type, video_title, target_frame=None):
    # frame_type denotes any subdirectory in find_dataset/ (apart from merged_maps)
    init_path = os.path.join("find_dataset", frame_type, video_title)

    frames = []
    for frame_index, frame_name in enumerate(sorted(os.listdir(init_path))):
        frame = cv2.imread(os.path.join(init_path, frame_name))

        if target_frame == frame_index:
            return frame

        frames.append(frame)

    return frames


def get_feature_frames(video_title, target_frame=None):
    init_path = os.path.join("find_dataset", "merged_maps", video_title)

    merged_frames = []
    speaker_frames = []

    # to store speaker and non-speaker frames separately
    temp_frames = {}
    for frame_name in sorted(os.listdir(init_path), key=_extract_frame_number):
        frame_number = _extract_frame_number(frame_name)
        is_nonspeaker_frame = "nonspeaker" in frame_name

        if not frame_number in temp_frames:
            temp_frames[frame_number] = {"speaker": None, "non_speaker": None}

        frame = cv2.imread(os.path.join(init_path, frame_name), cv2.IMREAD_GRAYSCALE)

        if is_nonspeaker_frame:
            temp_frames[frame_number]["non_speaker"] = frame
        else:
            temp_frames[frame_number]["speaker"] = frame

        if target_frame is not None and target_frame == frame_number:
            merged_frame, one_hot_array = _process_frame(temp_frames[frame_number])
            return merged_frame, one_hot_array

    for frame_number, frames in temp_frames.items():
        merged_frame, speaker_frame = _process_frame(frames)
        merged_frames.append(merged_frame)
        speaker_frames.append(speaker_frame)

    return merged_frames, speaker_frames


def _process_frame(frames):
    # merge speaker and non-speaker frames using bitwise 'or'
    merged_frame = cv2.bitwise_or(frames["speaker"], frames["non_speaker"])
    # set to white all pixels that correspond exclusively to the speaker
    speaker_frame = (frames["speaker"] == 255).astype(np.uint8)

    return merged_frame, speaker_frame


def _extract_frame_number(filename):
    match = re.search(r"_f(\d+)_", filename)
    # if the pattern isn't found, there's somethign wrong with the naming
    # i'll assume that's not the case (the names were probably generated automatically, anyhow)
    return int(match.group(1))
