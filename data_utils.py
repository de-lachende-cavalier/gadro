import os
import re
import cv2


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


def get_merged_maps(video_title, target_frame=None, speaker=True, non_speaker=True):
    init_path = os.path.join("find_dataset", "merged_maps", video_title)

    frames = []
    speaker_info = []
    for frame_name in sorted(os.listdir(init_path), key=_extract_frame_number):
        is_nonspeaker_frame = "nonspeaker" in frame_name

        if (non_speaker and is_nonspeaker_frame) or (
            speaker and not is_nonspeaker_frame
        ):
            frame = cv2.imread(os.path.join(init_path, frame_name))

            if target_frame == _extract_frame_number(frame_name):
                return frame, [not is_nonspeaker_frame]

            frames.append(frame)
            speaker_info.append(not is_nonspeaker_frame)

    return frames, speaker_info


def _extract_frame_number(filename):
    match = re.search(r"_f(\d+)_", filename)
    # if the pattern isn't found, there's somethign wrong with the naming
    # i'll assume that's not the case (the names were probably generated automatically, anyhow)
    return int(match.group(1))
