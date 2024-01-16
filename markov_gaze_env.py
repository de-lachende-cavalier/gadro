import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from preprocess_utils import compute_frame_features, compute_foa_features


class MarkovGazeEnv(gym.Env):
    """A Markovian gaze environment, i.e., one in which the next FoA (focus of attention) only depends on the current frame.

    One doesn't expect just one frame to suffice for correct prediction, but this environment is mostly meant to serve as a useful baseline.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # we create one env per video AND per subject
    def __init__(
        self,
        patch_bounding_boxes,
        patch_centres,
        speaker_info,
        foa_centres,
        patch_weights_per_frame,
        frame_idx=0,
        frame_width=1280,
        frame_height=720,
        render_mode=None,
    ):
        self.patch_bounding_boxes_all_frames = patch_bounding_boxes
        self.patch_centres_all_frames = patch_centres
        self.speaker_info_all_frames = speaker_info
        self.foa_centres_all_frames = foa_centres
        self.patch_weights_all_frames = patch_weights_per_frame

        self.current_frame_idx = frame_idx
        self.frame_width = frame_width
        self.frame_height = frame_height

        num_patches = len(
            self.patch_centres_all_frames[0]
        )  # they're the same number across frames

        # each observation corresponds to all the details about ONE frame (the env is markovian, after all)
        self.observation_space = spaces.Dict(
            {
                "patch_centres": spaces.Box(
                    low=np.zeros((num_patches, 2)),
                    high=np.array(
                        [self.frame_width, self.frame_height] * num_patches
                    ).reshape(num_patches, 2),
                    shape=(num_patches, 2),
                    dtype=np.float32,
                ),
                "patch_bounding_boxes": spaces.Box(
                    low=np.zeros((num_patches, 4)),
                    high=np.array(
                        [
                            self.frame_width,
                            self.frame_height,
                            self.frame_width,
                            self.frame_height,
                        ]
                        * num_patches
                    ).reshape(num_patches, 4),
                    shape=(num_patches, 4),
                    dtype=np.float32,
                ),
                "speaker_info": spaces.MultiBinary(num_patches),
                # below is the "label"
                "attended_patch_idx": spaces.Discrete(num_patches),
            }
        )
        # choose which patch to attend to, by choosing the correct index to use
        self.action_space = spaces.Discrete(num_patches)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self):
        """Selects the right frame to present as an observation, given the information regarding alle the frames."""

        observation = {
            "patch_centres": self.patch_centres_all_frames[self.current_frame_idx],
            "patch_bounding_boxes": self.patch_bounding_boxes_all_frames[
                self.current_frame_idx
            ],
            "speaker_info": self.speaker_info_all_frames[self.current_frame_idx],
            "current_attention": self.foa_centres_all_frames[self.current_frame_idx],
        }
        return observation

    def _get_info(self):
        # TODO what could I put here?
        info = {}
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_frame_idx = 0  # reset the frame counter
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        chosen_patch_centre = self.patch_centres_all_frames[self.current_frame_idx][
            action
        ]

        attention_weight = self.patch_weights_all_frames[self.current_frame_idx][
            chosen_patch_centre
        ]

        self.current_frame_idx += 1
        observation = self._get_observation()
        info = self._get_info()

        correct_guess = observation["current_attention"] == chosen_patch_centre
        # TODO some more reward engineering!
        reward = attention_weight if correct_guess else 0

        terminated = self.current_frame_idx >= len(self.patch_bounding_boxes_all_frames)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.img_width, self.img_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        image_frame = self._get_current_image_frame()
        canvas = pygame.surfarray.make_surface(
            np.transpose(image_frame, axes=(1, 0, 2))
        )

        for index, bounding_box in enumerate(self.all_patch_bounding_boxes):
            x, y, width, height = bounding_box
            patch_color = (
                (0, 255, 0) if self.current_attention[index] else (255, 0, 0)
            )  # green if attended by human, red otherwise

            pygame.draw.rect(
                canvas, patch_color, pygame.Rect(x, y, width, height), width=3
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _get_current_image_frame(self):
        return np.zeros(
            (self.img_width, self.img_height, 3), dtype=np.uint8
        )  # placeholder

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# impromptu testing code
if __name__ == "__main__":

    def test_initialisation(env):
        assert env.frame_width == 1280
        assert env.frame_height == 720
        assert env.observation_space != None
        assert env.action_space != None

        print("test_initialisation passed!")

    def test_reset(env):
        observation, info = env.reset()
        assert env.current_frame_idx == 0
        assert "patch_centres" in observation

        print("test_reset passed!")

    def test_step(env):
        env.reset()
        action = 0  # Assuming this is a valid action
        observation, reward, terminated, _, info = env.step(action)

        assert env.current_frame_idx == 1
        assert reward != None
        assert "patch_centres" in observation

        print("test_step passed!")

    def test_close(env):
        env.close()
        assert env.window == None

        print("test_close passed!")

    # use just two, hand-crafted, frames
    patch_bounding_boxes = [[(0, 0, 100, 100)]] * 2
    patch_centres = [[(50, 50)]] * 2
    speaker_info = [[True]] * 2
    foa_centres = [[(50, 50)]] * 2
    patch_weights_per_frame = [
        {center: 1.0 for center in foa_centres[i]} for i in range(2)
    ]

    env = MarkovGazeEnv(
        patch_bounding_boxes,
        patch_centres,
        speaker_info,
        foa_centres,
        patch_weights_per_frame,
    )

    test_initialisation(env)
    test_reset(env)
    test_step(env)
    test_close(env)

    # let's test thigs with actual data
    vid_filename = "012"
    patch_bounding_boxes, patch_centres, speaker_info = compute_frame_features(
        vid_filename
    )

    foa_centres, patch_weights_per_frame = compute_foa_features(
        vid_filename + ".mat", patch_centres
    )
    # the foa_centres are for all subjects, but the env is supposed to work only for one
    target_subject = 0
    foa_centres_single_subject = [frame[target_subject] for frame in foa_centres]

    env = MarkovGazeEnv(
        patch_bounding_boxes,
        patch_centres,
        speaker_info,
        foa_centres_single_subject,
        patch_weights_per_frame,
    )

    print("\n[+] Tests with real data:\n")

    test_initialisation(env)
    test_reset(env)
    test_step(env)
    test_close(env)
