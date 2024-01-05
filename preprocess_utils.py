import cv2
import numpy as np

from scipy.io import loadmat
from scipy.spatial.distance import euclidean

from data_utils import get_merged_maps


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
        patch_centres, key=lambda p: euclidean(gaze_point, (p[0], p[1]))
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
if __name__ == "__main__":

    def test_detect_patches():
        # a simple binary image with known patches
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (21, 21), (40, 40), (255, 255, 255), -1)
        cv2.rectangle(test_image, (61, 61), (80, 80), (255, 255, 255), -1)

        _, patch_centers = detect_patches(test_image)
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
        closest_patch = find_closest_patch(gaze_point, patch_centers)

        assert closest_patch == (30, 30), "Failed to find the correct closest patch."

        print("test_find_closest_patch passed!")

    def test_preprocess_mat_data(file_name, patch_centers):
        weights = preprocess_mat_data(file_name, patch_centers)

        # check if weights sum to 1 for each frame
        for frame_idx in range(600):
            total_weight = sum(weights[frame_idx].values())
            assert np.isclose(
                total_weight, 1
            ), f"Weights do not sum to 1 for frame {frame_idx}."

        print("test_preprocess_mat_data passed!")

    test_detect_patches()
    test_find_closest_patch()

    image, speaker_info = get_merged_maps(
        "012", target_frame=0, speaker=False, non_speaker=True
    )
    assert sum(speaker_info) == 0  # should be False
    bounding_rects, patch_centers = detect_patches(image)  # should return three patches
    assert len(bounding_rects) == 3

    test_preprocess_mat_data("001.mat", patch_centers)
