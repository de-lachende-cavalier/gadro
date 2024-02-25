import numpy as np
import random

from env_frames import FramesEnvironment


class FlatFramesEnvironment(FramesEnvironment):
    """A variant of the Frames Environment, in which we flatten our observations instead of keeping them in dictionary form.

    This environment mostly exists because of Tianshou API constraints.
    """

    def step(self, action):
        self._observed_frames.add(self.current_frame_idx)
        reward_index = self.current_frame_idx + 1
        chosen_patch_centre = self.patch_centres_all_frames[reward_index][action]
        attention_weight = self.patch_weights_all_frames[reward_index][
            chosen_patch_centre
        ]

        reward_observation = self._get_observation(reward_index, time_horizon=1)
        # the attended_patch_centres are the first two values in the flattened observatoin
        correct_guess = (reward_observation[:2] == chosen_patch_centre).all()

        reward = (1 + attention_weight) if correct_guess else 0

        if reward_index == self._num_frames - 1:
            self._observed_frames.add(reward_index)
        unseen_frames = list(set(range(self._num_frames - 1)) - self._observed_frames)
        self.current_frame_idx = random.choice(unseen_frames) if unseen_frames else 0

        observation = self._get_observation(self.current_frame_idx)
        info = self._get_info()

        terminated = len(self._observed_frames) == self._num_frames

        self._last_action = action
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def unflatten(self, flattened_obs):
        info = self._get_info()
        sorted_keys = info["sorted_keys"]
        shapes = info["shapes"]

        unflattened_obs = {}
        start = 0
        for key in sorted_keys:
            end = start + np.prod(shapes[key])

            # reshape the relevant slice of the flattened array to the original shape
            unflattened_obs[key] = flattened_obs[start:end].reshape(shapes[key])

            start = end

        return unflattened_obs

    def _get_observation(self, frame_index, time_horizon=None):
        observation = super()._get_observation(frame_index, time_horizon)

        # flatten the observation dict, ensuring keys are alphabetically sorted
        flattened_observation = self._flatten_and_sort_dict(observation)

        return flattened_observation

    def _flatten_and_sort_dict(self, observation):
        sorted_keys = sorted(observation.keys())

        flattened = np.concatenate(
            [np.array(observation[key]).flatten() for key in sorted_keys]
        )

        return flattened

    def _get_info(self):
        info = super()._get_info()

        sample_obs = super()._get_observation(0)
        shapes = {key: np.array(value).shape for key, value in sample_obs.items()}

        info["sorted_keys"] = sorted(sample_obs.keys())
        info["shapes"] = shapes

        return info
