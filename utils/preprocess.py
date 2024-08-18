"""
This Python file contains utilities for preprocessing the data, before feeding it to the RL agent.

The functions whose name begins with an underscore are considered "private", i.e., only meant for internal use within this file. They're undocumented because they're quite self explanatory and short.

Functions:
    compute_frame_features - Returns all the useful information (features) extracted from the frames containing the merged maps, for a single video: the bounding boxes of the various patches, their centers, and a boolean array denoting which patch corresponds to the speaker. 

    compute_foa_features - Returns all the useful information (features) extracted from the MATLAB files, for a single video and all subjects: a list of lists containing the FoAs of each subject at each frame and a list of lists containing some weights, quantifying, roughly, the probability that any of the 39 subjects pays attention to some FoA at some time t.
"""

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

from data import get_feature_frames, get_mat_data


def compute_frame_features(vid_filename):
    merged_frames, speaker_frames = get_feature_frames(vid_filename)

    # an array of bounding boxes, indexed by frame number
    bounding_boxes = []
    # an array of patch centers, indexed by frame number
    patch_centres = []
    # an array of boolean arrays, indicating who is(are) the speaker(s) for each frame
    speaker_info = []
    for frame, speak_frame in zip(merged_frames, speaker_frames):
        boxes_merged, centres_merged = _detect_patches(frame)
        boxes_speak, centres_speak = _detect_patches(speak_frame)

        speaker_info.append([centre in centres_speak for centre in centres_merged])
        patch_centres.append(centres_merged)
        bounding_boxes.append(boxes_merged)

    return bounding_boxes, patch_centres, speaker_info


def compute_foa_features(mat_filename, patch_centres):
    fix_data, num_subjects, num_frames = get_mat_data(mat_filename)

    foa_centres = [
        {subject: (0, 0) for subject in range(num_subjects)} for _ in range(num_frames)
    ]
    # we will assign a weight to each patch (for each frame), depending on the number of subjects that were observing it, this is done as a form of basic feature engineering
    frame_patch_weights = [
        {center: 0 for center in patch_centres[i]} for i in range(num_frames)
    ]

    for subject_idx in range(num_subjects):
        subject_data = fix_data[subject_idx][0]

        for frame_idx in range(num_frames):
            gaze_point = subject_data[frame_idx]
            closest_patch_centre = _find_closest_patch(
                gaze_point, patch_centres[frame_idx]
            )
            frame_patch_weights[frame_idx][closest_patch_centre] += 1

            # normalise
            total_gazes = sum(frame_patch_weights[frame_idx].values())
            for patch in frame_patch_weights[frame_idx]:
                frame_patch_weights[frame_idx][patch] /= total_gazes

            foa_centres[frame_idx][subject_idx] = closest_patch_centre

    return foa_centres, frame_patch_weights


def _detect_patches(image):
    if len(image.shape) == 2:
        # in grayscale, no further processing needed
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_rects, patch_centers = [], []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width / 2
        center_y = y + height / 2

        patch_centers.append((center_x, center_y))
        bounding_rects.append(cv2.boundingRect(contour))

    return bounding_rects, patch_centers


def _find_closest_patch(gaze_point, patch_centres):
    closest_patch_centre = min(
        patch_centres, key=lambda p: euclidean(gaze_point, (p[0], p[1]))
    )
    return closest_patch_centre


# impromptu testing environment
if __name__ == "__main__":

    def test_compute_frame_features(vid_filename):
        bounding_boxes, patch_centres, speaker_info = compute_frame_features(
            vid_filename
        )

        merged_frames, speaker_frames = get_feature_frames(vid_filename)
        num_frames = len(merged_frames)

        assert len(bounding_boxes) == num_frames, "Bounding boxes length mismatch"
        assert len(patch_centres) == num_frames, "Patch centres length mismatch"
        assert len(speaker_info) == num_frames, "Speaker info length mismatch"

        for boxes, centres, info in zip(bounding_boxes, patch_centres, speaker_info):
            assert isinstance(boxes, list), "Bounding boxes are not in list format"
            assert isinstance(centres, list), "Patch centres are not in list format"
            assert isinstance(info, list), "Speaker info is not in list format"
            assert all(
                isinstance(centre, tuple) for centre in centres
            ), "Patch centres are not tuples"

        for frame_idx, (frame, speak_frame) in enumerate(
            zip(merged_frames, speaker_frames)
        ):
            _, centres_merged = _detect_patches(frame)
            _, centres_speak = _detect_patches(speak_frame)
            for centre in centres_merged:
                assert (centre in centres_speak) == speaker_info[frame_idx][
                    centres_merged.index(centre)
                ], "Speaker info accuracy mismatch"

        print("test_compute_frame_features passed!")

    def test_detect_patches():
        # a simple binary image with known patches
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (21, 21), (40, 40), (255, 255, 255), -1)
        cv2.rectangle(test_image, (61, 61), (80, 80), (255, 255, 255), -1)

        _, patch_centers = _detect_patches(test_image)
        assert (
            31,
            31,
        ) in patch_centers, "Failed to detect the center of the first patch."
        assert (
            71,
            71,
        ) in patch_centers, "Failed to detect the center of the second patch."

        print("test_detect_patches passed!")

    def test_find_closest_patch():
        patch_centers = [(30, 30), (70, 70)]
        gaze_point = (35, 35)
        closest_patch = _find_closest_patch(gaze_point, patch_centers)

        assert closest_patch == (30, 30), "Failed to find the correct closest patch."

        print("test_find_closest_patch passed!")

    def test_compute_patch_weights(mat_filename, patch_centers):
        _, weights = compute_foa_features(mat_filename, patch_centers)

        _, _, num_frames = get_mat_data(mat_filename)
        # check if weights sum to 1 for each frame
        for frame_idx in range(num_frames):
            total_weight = sum(weights[frame_idx].values())
            assert np.isclose(
                total_weight, 1
            ), f"Weights do not sum to 1 for frame {frame_idx}."

        print("test_compute_patch_weights passed!")

    test_detect_patches()
    test_find_closest_patch()

    vid_filename = "012"
    test_compute_frame_features(vid_filename)

    boxes, centres, speaker_info = compute_frame_features("012")
    # there are four people in the video
    assert all(len(box) == 4 for box in boxes)
    assert all(len(centre) == 4 for centre in centres)

    # there's almost alwyas one speaker, apart from the final few seconds
    assert all(sum(sinfo) <= 2 for sinfo in speaker_info)

    test_compute_patch_weights("012.mat", centres)
