import cv2
import numpy as np

from scipy.io import loadmat
from scipy.spatial.distance import euclidean

from data_utils import merge_and_encode_features


def preprocess_frame_features(filename):
    merged_frames, speaker_frames = merge_and_encode_features(filename)

    # an array of bounding boxes, indexed by frame number
    bounding_boxes = []
    # an array of patch centers, indexed by frame number
    patch_centres = []
    # an array of boolean arrays, indicating who is(are) the speaker(s) for each frame
    speaker_info = []
    for frame, speak_frame in zip(merged_frames, speaker_frames):
        boxes_merged, centres_merged = _detect_patches(frame)
        boxes_speak, centres_speak = _detect_patches(speak_frame)

        speaker_info.append(
            [
                (bs, cs) in zip(boxes_merged, centres_merged)
                for bs, cs in zip(boxes_speak, centres_speak)
            ]
        )
        patch_centres.append(centres_merged)
        bounding_boxes.append(boxes_merged)

    return bounding_boxes, patch_centres, speaker_info


# TODO the patch centres are not as I expect!! => before I only had them for one frame, now I have them for all frames ==> they're a list of lists of tuples!! => problem, cause lists are unhashable
def preprocess_mat_data(filename, patch_centres):
    base = "find_dataset/Our_database/fix_data_NEW/"

    mat_data = loadmat(base + filename)
    # (39, 1) nparray, one entry per subject
    # to get the actual data do fix_data[subject_idx][0] => (600, 2)
    fix_data = mat_data["curr_v_all_s"]

    num_subjects = fix_data.shape[0]
    num_frames = 600

    # we will assign a weight to each patch, dependeing on the number of subjects that were observing it at some time t => the bigger the number of subjects observing it, the higher the weight
    # this same weight will influence reward: if the agent chooses a patch with high weight, it gets a high reward, more or less
    # considering that we're normalising things, these could very well be regarded as probabilities that some "average human" pays attention to some patch
    attended_patch_centres = []
    frame_patch_weights = [
        {center: 0 for center in patch_centres} for _ in range(num_frames)
    ]
    for subject_idx in range(num_subjects):
        subject_data = fix_data[subject_idx][0]

        for frame_idx in range(num_frames):
            gaze_point = subject_data[frame_idx]
            closest_patch_centre = _find_closest_patch(gaze_point, patch_centres)
            # the patch that is attended to is the one closest to the gaze point
            attended_patch_centres.append(closest_patch_centre)

            frame_patch_weights[frame_idx][closest_patch_centre] += 1

    # normalise
    for frame_idx in range(num_frames):
        total_gazes = sum(frame_patch_weights[frame_idx].values())
        for patch in frame_patch_weights[frame_idx]:
            frame_patch_weights[frame_idx][patch] /= total_gazes

    return attended_patch_centres, frame_patch_weights


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

    def test_preprocess_mat_data(file_name, patch_centers):
        _, weights = preprocess_mat_data(file_name, patch_centers)

        # check if weights sum to 1 for each frame
        for frame_idx in range(600):
            total_weight = sum(weights[frame_idx].values())
            assert np.isclose(
                total_weight, 1
            ), f"Weights do not sum to 1 for frame {frame_idx}."

        print("test_preprocess_mat_data passed!")

    test_detect_patches()
    test_find_closest_patch()

    boxes, centres, speaker_info = preprocess_frame_features("012")
    # there are four people in the video
    assert all(len(box) == 4 for box in boxes)
    assert all(len(centre) == 4 for centre in centres)

    # there's almost alwyas one speaker, apart from the final few seconds
    assert all(sum(sinfo) <= 2 for sinfo in speaker_info)

    test_preprocess_mat_data("012.mat", centres)
