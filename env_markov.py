import numpy as np
import random
import pygame

import gymnasium as gym
from gymnasium import spaces

from utils_preprocess import compute_frame_features, compute_foa_features


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
        frame_width=1280,
        frame_height=720,
        render_mode=None,
    ):
        self.patch_bounding_boxes_all_frames = patch_bounding_boxes
        self.patch_centres_all_frames = patch_centres
        self.speaker_info_all_frames = speaker_info
        self.foa_centres_all_frames = foa_centres
        self.patch_weights_all_frames = patch_weights_per_frame

        self.frame_width = frame_width
        self.frame_height = frame_height

        # set of ints, containing indices of observed frames
        self._observed_frames = set()
        # useful for rendering, store the last action taken
        self._last_action = None
        # useful in the step method
        self._num_frames = len(self.patch_bounding_boxes_all_frames)

        self._num_patches = len(
            self.patch_centres_all_frames[0]
        )  # they're the same number across frames

        # each observation corresponds to all the details about ONE frame (the env is markovian, after all)
        self.observation_space = spaces.Dict(
            {
                "patch_centres": spaces.Box(
                    low=np.zeros((self._num_patches, 2)),
                    high=np.array(
                        [self.frame_width, self.frame_height] * self._num_patches
                    ).reshape(self._num_patches, 2),
                    shape=(self._num_patches, 2),
                    dtype=np.float32,
                ),
                "patch_bounding_boxes": spaces.Box(
                    low=np.zeros((self._num_patches, 4)),
                    high=np.array(
                        [
                            self.frame_width,
                            self.frame_height,
                            self.frame_width,
                            self.frame_height,
                        ]
                        * self._num_patches
                    ).reshape(self._num_patches, 4),
                    shape=(self._num_patches, 4),
                    dtype=np.float32,
                ),
                "speaker_info": spaces.MultiBinary(self._num_patches),
                # below is the "label"
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
        observation = {
            "patch_centres": self.patch_centres_all_frames[frame_index],
            "patch_bounding_boxes": self.patch_bounding_boxes_all_frames[frame_index],
            "speaker_info": self.speaker_info_all_frames[frame_index],
            "attended_patch_centre": self.foa_centres_all_frames[frame_index],
        }

        return observation

    def _get_info(self):
        # TODO what could I put here?
        info = {}
        return info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # select a random index to start from (but never the last one!)
        self.current_frame_idx = random.choice(range(self._num_frames - 1))

        # initialise the set
        self._observed_frames = set()

        observation = self._get_observation(self.current_frame_idx)
        info = self._get_info()

        self._last_action = None
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # this needs to be done right away, and only for the current frame
        self._observed_frames.add(self.current_frame_idx)

        # REWARD CALCULATION

        # we can safely increment the frame index because we're sure that we'll get an index that's in the correct range
        reward_index = self.current_frame_idx + 1

        chosen_patch_centre = self.patch_centres_all_frames[reward_index][action]

        attention_weight = self.patch_weights_all_frames[reward_index][
            chosen_patch_centre
        ]

        reward_observation = self._get_observation(reward_index)
        correct_guess = (
            reward_observation["attended_patch_centre"] == chosen_patch_centre
        )

        # reward the agent for picking the correct patch
        # add a bonus based on how likely it is that the patch is picked by other subjects to push it a bit towards generalisation
        reward = (1 + attention_weight) if correct_guess else 0

        # STATE MANAGEMENT

        # only add the reward frame to the observed ones if it's the final one
        if reward_index == self._num_frames - 1:
            # we need to do it manually (we can never choose it, so we'd never get to add it otherwise)
            self._observed_frames.add(reward_index)

        # we now need to choose the next frame
        unseen_frames = list(
            set(range(self._num_frames - 1)) - self._observed_frames
        )  # we need to exclude the last frame from the selection process!

        # we return a default 0 if unseen_frames is empty (i.e., if we need to terminate an episode), to avoid annoying errors
        self.current_frame_idx = random.choice(unseen_frames) if unseen_frames else 0

        # get the actual next observation (the one the agent uses to act in the environment)
        observation = self._get_observation(self.current_frame_idx)
        info = self._get_info()

        # if you've observed all the frames in the environment once, terminate the episode
        terminated = len(self._observed_frames) == self._num_frames

        self._last_action = action
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
            self.window = pygame.display.set_mode((self.frame_width, self.frame_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        image_frame = self._get_current_image_frame()
        canvas = pygame.surfarray.make_surface(
            np.transpose(image_frame, axes=(1, 0, 2))
        )

        for index, bounding_box in enumerate(
            self.patch_bounding_boxes_all_frames[self.current_frame_idx]
        ):
            x, y, width, height = bounding_box
            human_color = (0, 255, 0)  # green for human attended patches
            agent_color = (0, 0, 255)  # blue for agent attended patches
            both_color = (255, 255, 0)  # yellow if both human and agent attend
            none_color = (255, 0, 0)  # red if attended by neither

            human_attends = self.speaker_info_all_frames[self.current_frame_idx][index]
            agent_attends = self._last_action == index

            if human_attends and agent_attends:
                patch_color = both_color
            elif human_attends:
                patch_color = human_color
            elif agent_attends:
                patch_color = agent_color
            else:
                patch_color = none_color

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
        # return a completely black frame
        return np.zeros((self.frame_width, self.frame_height, 3), dtype=np.uint8)

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

        assert env._observed_frames == set()
        assert env.current_frame_idx in range(env._num_frames - 1)
        assert "patch_centres" in observation

        print("test_reset passed!")

    def test_step(env):
        env.reset()
        action = 0  # Assuming this is a valid action
        observation, reward, terminated, _, info = env.step(action)

        assert env.current_frame_idx in range(env._num_frames - 1)
        assert reward != None
        assert "patch_centres" in observation

        print("test_step passed!")

    def test_close(env):
        env.close()
        assert env.window == None

        print("test_close passed!")

    def test_random_frame_selection(env):
        env.reset()
        seen_frames = set()
        for _ in range(100):  # arbitrary number
            env.step(env.action_space.sample())  # random action
            assert env.current_frame_idx not in seen_frames
            seen_frames.add(env.current_frame_idx)
        print("test_random_frame_selection passed!")

    def test_termination_condition(env):
        env.reset()
        max_steps = (
            len(env.patch_bounding_boxes_all_frames) * 2
        )  # twice the number of frames as steps should suffice
        step_count = 0

        while step_count < max_steps:
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            step_count += 1

            if terminated:
                break

        assert len(env._observed_frames) == len(env.patch_bounding_boxes_all_frames)
        print("test_termination_condition passed!")

    def test_render_method(env):
        env.reset()
        try:
            env.render()
            print("test_render_method passed!")
        except Exception as e:
            print(f"test_render_method failed: {e}")

    def test_reward_calculation(
        env, patch_centres, patch_weights, foa_centres_single_subject
    ):
        env.reset()

        while True:
            cur_index = env.current_frame_idx

            action = env.action_space.sample()
            _, reward, terminated, _, _ = env.step(action)

            chosen_patch_centre = patch_centres[cur_index + 1][action]
            attention_weight = patch_weights[cur_index + 1][chosen_patch_centre]
            correct_guess = (
                foa_centres_single_subject[cur_index + 1] == chosen_patch_centre
            )

            reward_manual = (1 + attention_weight) if correct_guess else 0
            # if reward calculation works, we can also be sure that the frame succession in the step() method works!
            assert reward_manual == reward

            if terminated:
                break

        print("test_reward_calculation passed!")

    # use just two hand-crafted frames
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

    test_random_frame_selection(env)
    test_render_method(env)
    test_termination_condition(env)

    test_reward_calculation(
        env, patch_centres, patch_weights_per_frame, foa_centres_single_subject
    )

    test_close(env)
