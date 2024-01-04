import cv2
import numpy as np

from scipy.io import loadmat
from scipy.spatial.distance import euclidean


def detect_patches(bitmap):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_BGR2GRAY)
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


def find_closest_patch(gaze_point, patch_centres):
    closest_patch = min(
        patch_centers, key=lambda p: euclidean(gaze_point, (p[0], p[1]))
    )
    return closest_patch


def preprocess_mat_data(filename, patch_centres):
    base = "find_dataset/Our_database/fix_data_NEW/"

    mat_data = loadmat(base + filename)
    # (39, 1) nparray, one entry per subject
    # to get the actual data do fix_data[subject_idx][0] => (600, 2)
    fix_data = mat_data["curr_v_all_s"]

    num_subjects = fix_data.shape[0]
    num_frames = 600

    # we will assign a weight to each patch, dependeing on the number of subjects that were observing it at some time t => the bigger the number of subject observing it, the higher the weight
    # this same weight will influence reward: if the agent chooses a patch with high weight, it gets a high reward, more or less
    # considering that we're normalising things, these could very well be regarded as probabilities that some "average human" pays attention to some patch
    # this allows us to still use the data from all 39 subjects as well! (how?)
    frame_patch_weights = [
        {center: 0 for center in patch_centers} for _ in range(num_frames)
    ]
    for subject_idx in range(num_subjects):
        subject_data = fix_data[subject_idx][0]

        for frame_idx in range(num_frames):
            gaze_point = subject_data[frame_idx]
            closest_patch = find_closest_patch(gaze_point, patch_centres)
            frame_patch_weights[frame_idx][closest_patch] += 1

    # normalise
    for frame_idx in range(num_frames):
        total_gazes = sum(frame_patch_weights[frame_idx].values())
        for patch in frame_patch_weights[frame_idx]:
            frame_patch_weights[frame_idx][patch] /= total_gazes

    return frame_patch_weights


# impromptu testing environment
# TODO better testing! also, clear things up with the preprocess function
if __name__ == "__main__":
    image = cv2.imread("find_dataset/merged_maps/012/012_f0_nonspeaker.png")
    bounding_rects, patch_centers = detect_patches(image)  # should return three patches

    weights = preprocess_mat_data("001.mat", patch_centers)
