import numpy as np

from gymnasium import spaces

from base import BaseEnvironment


class FramesEnvironment(BaseEnvironment):
    """An evolution on the base environment, in which we also includes all the frame varieties we have at our disposal.

    One would expect this extra information to improve performance (and, naturally, make training more computationally demanding).
    """

    def __init__(
        self,
        time_horizon,
        video_frames,
        dynamics_frames,
        patches_frames,
        patch_bounding_boxes,
        patch_centres,
        speaker_info,
        foa_centres,
        patch_weights_per_frame,
        render_mode=None,
    ):
        self.T = time_horizon

        self.video_frames = video_frames
        self.dynamics_frames = dynamics_frames
        self.patches_frames = patches_frames

        # useful for rendering
        self.patch_bounding_boxes_all_frames = patch_bounding_boxes

        self.patch_centres_all_frames = patch_centres
        self.speaker_info_all_frames = speaker_info
        self.foa_centres_all_frames = foa_centres
        self.patch_weights_all_frames = patch_weights_per_frame

        # set of ints, containing indices of observed frames
        self._observed_frames = set()
        # useful for rendering, store the last action taken
        self._last_action = None
        # useful in the step method
        self._num_frames = len(self.video_frames)

        self._num_patches = len(
            self.patch_centres_all_frames[0]
        )  # they're the same number across frames

        self.frame_height = video_frames[0].shape[0]
        self.frame_width = video_frames[0].shape[1]

        self.observation_space = spaces.Dict(
            {
                "video_frames": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.T, self.frame_height, self.frame_width, 3),
                    dtype=np.uint8,
                ),
                "dynamics_frames": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.T, self.frame_height, self.frame_width, 3),
                    dtype=np.uint8,
                ),
                "patches_frames": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.T, self.frame_height, self.frame_width),
                    dtype=np.uint8,
                ),
                "patch_centres": spaces.Box(
                    low=np.zeros((self.T, self._num_patches, 2)),
                    high=np.tile(
                        np.array([self.frame_width, self.frame_height]).reshape(
                            1, 1, 2
                        ),
                        (self.T, self._num_patches, 1),
                    ),
                    dtype=np.float32,
                ),
                "speaker_info": spaces.MultiBinary((self.T, self._num_patches)),
                # Adjusted the "label" to include time dimension if necessary
                "attended_patch_centres": spaces.Box(
                    low=np.zeros((self.T, 2)),
                    high=np.tile(
                        np.array([self.frame_width, self.frame_height]).reshape(1, 2),
                        (self.T, 1),
                    ),
                    dtype=np.float32,
                ),
            }
        )

        # choose which patch to attend to, by choosing the correct index to use
        self.action_space = spaces.Discrete(self._num_patches)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self, frame_index, time_horizon=None):
        if time_horizon is None:
            time_horizon = self.T

        start_frame_index = max(0, frame_index - time_horizon + 1)
        end_frame_index = frame_index + 1

        observation = {
            "video_frames": [],
            "dynamics_frames": [],
            "patches_frames": [],
            "patch_centres": [],
            "speaker_info": [],
            "attended_patch_centres": [],
        }

        # check if we need to pad the observation due to insufficient history
        num_missing_frames = time_horizon - (end_frame_index - start_frame_index)
        if num_missing_frames > 0:
            # use the earliest available frame to pad the missing history
            earliest_frame = {
                "video_frames": self.video_frames[start_frame_index],
                "dynamics_frames": self.dynamics_frames[start_frame_index],
                "patches_frames": self.patches_frames[start_frame_index],
                "patch_centres": self.patch_centres_all_frames[start_frame_index],
                "speaker_info": self.speaker_info_all_frames[start_frame_index],
                "attended_patch_centres": self.foa_centres_all_frames[
                    start_frame_index
                ],
            }
            for _ in range(num_missing_frames):
                for key in observation:
                    observation[key].append(earliest_frame[key])

        for frame_idx in range(start_frame_index, end_frame_index):
            current_frame = {
                "video_frames": self.video_frames[frame_idx],
                "dynamics_frames": self.dynamics_frames[frame_idx],
                "patches_frames": self.patches_frames[frame_idx],
                "patch_centres": self.patch_centres_all_frames[frame_idx],
                "speaker_info": self.speaker_info_all_frames[frame_idx],
                "attended_patch_centres": self.foa_centres_all_frames[frame_idx],
            }
            for key in observation:
                observation[key].append(current_frame[key])

        # if time horizon is 1, simplify the structure by removing the unnecessary temporal dimension
        if time_horizon == 1:
            for key in observation:
                observation[key] = observation[key][0]

        return observation

    # for rendering
    def _get_current_image_frame(self):
        return self.video_frames[self.current_frame_idx]
