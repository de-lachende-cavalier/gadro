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


def get_frames_by_type(frame_type, video_title):
    # frame_type denotes any subdirectory in find_dataset/ (apart from merged_maps)
    init_path = os.path.join("find_dataset", frame_type, video_title)

    frames = []
    for frame_name in sorted(os.listdir(init_path)):
        frame = cv2.imread(os.path.join(init_path, frame_name))

        frames.append(frame)

    return frames


def get_feature_frames(video_title):
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

    for frame_number, frames in temp_frames.items():
        # merge speaker and non-speaker frames using bitwise 'or'
        merged_frame = cv2.bitwise_or(frames["speaker"], frames["non_speaker"])
        merged_frames.append(merged_frame)

        speaker_frames.append(frames["speaker"])

    return merged_frames, speaker_frames


def _extract_frame_number(filename):
    match = re.search(r"_f(\d+)_", filename)
    # if the pattern isn't found, there's somethign wrong with the naming
    # i'll assume that's not the case for easier-to-read code
    return int(match.group(1))


if __name__ == "__main__":

    def test_get_mat_data(test_filename):
        fix_data, num_subjects, num_frames = get_mat_data(test_filename)

        assert isinstance(fix_data, np.ndarray), "fix_data is not a numpy ndarray"
        assert fix_data.shape == (39, 1), "fix_data shape is not (39, 1)"

        assert isinstance(num_subjects, int), "num_subjects is not an integer"
        assert (
            num_subjects == fix_data.shape[0]
        ), "num_subjects does not match fix_data shape"

        assert isinstance(num_frames, int), "num_frames is not an integer"
        assert (
            num_frames == fix_data[0][0].shape[0]
        ), "num_frames does not match the first subject's frame count"

        print("test_get_mat_data passed!")

    def test_get_frames_by_type(test_video_title, test_frame_type):
        frames = get_frames_by_type(test_frame_type, test_video_title)

        init_path = os.path.join("find_dataset", test_frame_type, test_video_title)
        expected_num_files = len(
            [
                name
                for name in os.listdir(init_path)
                if os.path.isfile(os.path.join(init_path, name))
            ]
        )

        assert isinstance(frames, list), "frames is not a list"

        for frame in frames:
            assert isinstance(frame, np.ndarray), "frame is not a numpy ndarray"

        assert (
            len(frames) == expected_num_files
        ), "Number of frames does not match number of files in directory"

        if frames:
            expected_shape = frames[0].shape
            for frame in frames:
                assert (
                    frame.shape == expected_shape
                ), "Not all frames have the same dimensions"

        print(f"test_get_frames_by_type passed! (type: {test_frame_type})")

    def _test_basic(video_title, show=True):
        merged_frames, speaker_frames = get_feature_frames(video_title)

        assert merged_frames, "No merged frames were returned"
        assert speaker_frames, "No speaker frames were returned"

        assert len(merged_frames) == len(speaker_frames)

        assert all(
            frame is not None for frame in speaker_frames
        ), "Empty speaker frames found"

        if show:
            for i, (merged_frame, speaker_frame) in enumerate(
                zip(merged_frames, speaker_frames)
            ):
                if i % 100 == 0:  # visualise every 100th frame pair
                    cv2.imshow(f"Merged Frame {i}", merged_frame)
                    cv2.imshow(f"Speaker Frame {i}", speaker_frame)
                    cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _test_merge_integrity(video_title):
        merged_frames, speaker_frames = get_feature_frames(video_title)

        for merged_frame, speaker_frame in zip(merged_frames, speaker_frames):
            non_speaker_frame = cv2.subtract(merged_frame, speaker_frame)
            expected_merged_frame = cv2.bitwise_or(speaker_frame, non_speaker_frame)

            assert np.array_equal(
                merged_frame, expected_merged_frame
            ), "Merged frame does not match expected bitwise 'or' result"

    def _test_frame_content(video_title):
        merged_frames, _ = get_feature_frames(video_title)

        for frame in merged_frames:
            assert np.any(frame), "Frame contains only default or empty values"

    def _test_white_patches_in_and_operation(video_title):
        merged_frames, speaker_frames = get_feature_frames(video_title)

        for merged_frame, speaker_frame in zip(merged_frames, speaker_frames):
            and_frame = cv2.bitwise_and(merged_frame, speaker_frame)
            assert np.any(
                and_frame == 255
            ), "No white patches found in bitwise AND operation"

    def test_get_feature_frames(video_title, show=True):
        _test_basic(video_title, show=show)
        _test_merge_integrity(video_title)
        _test_frame_content(video_title)
        _test_white_patches_in_and_operation(video_title)

        print("test_get_feature_frames passed!")

    test_video_title = "012"

    test_get_mat_data(test_video_title + ".mat")
    test_get_frames_by_type(test_video_title, "frames")
    test_get_frames_by_type(test_video_title, "dynamic")
    test_get_feature_frames(test_video_title, show=False)
