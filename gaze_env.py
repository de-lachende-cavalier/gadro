import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, all_patch_coordinates, render_mode=None):
        assert len(all_patch_coordinates) > 0, "Patch coordinates list cannot be empty."

        self.all_patch_coordinates = all_patch_coordinates
        num_patches = len(all_patch_coordinates)

        # observation space: bool array indicating attention on each patch
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_patches,), dtype=np.int32
        )

        # action space: selecting a patch from all available patches
        self.action_space = spaces.Discrete(num_patches)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.current_attention = None  # attention flags for current patches

    def _get_observation(self):
        # TODO
        return np.array(self.current_attention, dtype=np.int32)

    def _get_info(self):
        # TODO
        info = {}
        return info

    def reset(self, current_patch_data):
        num_patches = len(current_patch_data)
        assert (
            num_patches <= self.max_patches
        ), "Number of patches exceeds maximum allowed."

        # dynamically adjust the action space based on the current number of patches
        self.action_space = spaces.Discrete(num_patches)
        self.current_patches = current_patch_data

        observation = np.zeros((self.max_patches, 3), dtype=np.int32)
        observation[:num_patches] = current_patch_data

        info = None
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action."

        # action is the index of the chosen patch
        chosen_patch = self.all_patch_coordinates[action]

        terminated = self.current_attention[action]
        reward = 1 if terminated else 0

        observation = self._get_observation()
        info = self._get_info()

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

        for index, patch in enumerate(self.all_patch_coordinates):
            x, y = patch
            patch_color = (
                (0, 255, 0) if self.current_attention[index] else (255, 0, 0)
            )  # green if attended by human, red otherwise

            pygame.draw.rect(
                canvas, patch_color, pygame.Rect(x, y, patch_size, patch_size), width=3
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
# TODO test it!
if __name__ == "__main__":
    pass
