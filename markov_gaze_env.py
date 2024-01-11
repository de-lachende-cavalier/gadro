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

    # we create one env per video AND per subject => the environment should be oblivious to this!
    # TODO!
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
        self.patch_bounding_boxes = patch_bounding_boxes
        self.patch_centres = patch_centres
        self.speaker_info = speaker_info
        self.foa_centres = foa_centres
        self.patch_weights_per_frame = patch_weights_per_frame

        self.current_frame_idx = frame_idx
        self.frame_width = frame_width
        self.frame_height = frame_height

        num_patches = len(self.patch_centres[0])
        self.observation_space = spaces.Dict(
            {
                "patch_centres": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.frame_width, self.frame_height]),
                    shape=(num_patches, 2),
                    dtype=np.float32,
                ),
                "patch_bounding_boxes": spaces.Box(
                    low=0,
                    high=max(self.frame_width, self.frame_height),
                    shape=(num_patches, 4),
                    dtype=np.float32,
                ),
                "speaker_info": spaces.MultiBinary(num_patches),
                "current_attention": spaces.Box(
                    low=0, high=1, shape=(num_patches,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(num_patches)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_observation(self):
        observation = {
            "patch_centres": self.patch_centres[self.current_frame_idx],
            "patch_bounding_boxes": self.patch_bounding_boxes[self.current_frame_idx],
            "speaker_info": self.speaker_info[self.current_frame_idx],
            "current_attention": self.current_attention,
        }
        return observation

    def _get_info(self):
        # TODO what could I put here?
        info = {}
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_frame_idx = 0  # reset the frame counter
        self.action_space = spaces.Discrete(len(self.current_attention))
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action."

        chosen_patch = self.all_patch_coordinates[action]

        correct_guess = self.current_attention[
            action
        ]  # True if the patch is attended by humans
        attention_weight = self.attention_weights[self.current_frame_idx][chosen_patch]
        reward = attention_weight if correct_guess else 0

        observation = self._get_observation()
        info = self._get_info()

        terminated = self.current_frame_idx >= 600  # 20s at 30 FPS

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

    def test_initialization(env):
        assert env is not None, "Environment not initialized."
        assert env.observation_space is not None, "Observation space not initialized."
        assert env.action_space is not None, "Action space not initialized."
        print("Initialization test passed.")

    def test_step_function(env):
        initial_observation, _ = env.reset()
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        assert observation is not None, "Observation not returned by step function."
        assert isinstance(reward, float), "Reward not returned as float."
        assert isinstance(done, bool), "Done flag not returned as bool."
        print("Step function test passed.")

    def test_reset_function(env):
        observation, _ = env.reset()
        assert observation is not None, "Observation not returned by reset function."
        print("Reset function test passed.")

    def test_render_function(env):
        try:
            env.render()
            print("Render function test passed.")
        except Exception as e:
            assert False, f"Render function threw an exception: {e}"

    vid_filename = "012"
    patch_bounding_boxes, patch_centres, speaker_info = compute_frame_features(
        vid_filename
    )
    foa_centres, patch_weights = compute_foa_features(
        vid_filename + ".mat", patch_centres
    )

    env = MarkovGazeEnv(
        patch_bounding_boxes, patch_centres, speaker_info, foa_centres, patch_weights
    )

    test_initialization(env)
    test_step_function(env)
    test_reset_function(env)
    test_render_function(env)

    env.close()
