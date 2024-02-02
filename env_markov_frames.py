import numpy as np

from gymnasium import spaces

from env_markov import MarkovGazeEnv


class MarkovGazeEnvWithFrames(MarkovGazeEnv):
    """A Markovian gaze environment, i.e., one in which the next FoA (focus of attention) only depends on the current frame, but with more information per observation as the original one."""

    def __init__(
        self,
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
                "video_frame": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.frame_height, self.frame_width, 3),
                    dtype=np.uint8,
                ),
                "dynamics_frame": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.frame_height, self.frame_width, 3),
                    dtype=np.uint8,
                ),
                "patches_frame": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.frame_height, self.frame_width),
                    dtype=np.uint8,
                ),
                "patch_centres": spaces.Box(
                    low=np.zeros((self._num_patches, 2)),
                    high=np.array(
                        [self.frame_width, self.frame_height] * self._num_patches
                    ).reshape(self._num_patches, 2),
                    shape=(self._num_patches, 2),
                    dtype=np.float32,
                ),
                "speaker_info": spaces.MultiBinary(self._num_patches),
                "attended_patch_centre": spaces.Box(
                    low=0.0, high=self.frame_width, shape=(1, 2), dtype=np.float32
                ),
            }
        )

        # choose which patch to attend to, by choosing the correct index to use
        self.action_space = spaces.Discrete(self._num_patches)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self, frame_index):
        video_frame = self.video_frames[frame_index]
        dynamics_frame = self.dynamics_frames[frame_index]
        patches_frame = self.patches_frames[frame_index]

        observation = {
            "video_frame": video_frame,
            "dynamics_frame": dynamics_frame,
            "patches_frame": patches_frame,
            "patch_centres": self.patch_centres_all_frames[frame_index],
            "speaker_info": self.speaker_info_all_frames[frame_index],
            "attended_patch_centre": self.foa_centres_all_frames[frame_index],
        }

        return observation

    def _get_current_image_frame(self):
        return self.video_frames[self.current_frame_idx]
