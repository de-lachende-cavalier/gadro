from env_frames import FramesEnvironment


class FramesTestEnvironment(FramesEnvironment):
    """A testing environment for the Frames Environment, just as we have one for the base case."""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start from the first frame
        # (we don't worry about a non-markovian setting: we'll just pad the observations to fit the desired time horizon anyhow)
        self.current_frame_idx = 0

        observation = self._get_observation(self.current_frame_idx)
        info = self._get_info()

        self._last_action = None
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.current_frame_idx = min(self.current_frame_idx + 1, self._num_frames - 1)

        chosen_patch_centre = self.patch_centres_all_frames[self.current_frame_idx][
            action
        ]

        reward_observation = self._get_observation(
            self.current_frame_idx, time_horizon=1
        )
        correct_guess = (
            reward_observation["attended_patch_centres"] == chosen_patch_centre
        )

        # we change the reward structure to more correctly check how many frames the agent got right (i.e., a reward of 130, means that it correctly predicted the FoAs of 130 frames)
        reward = 1 if correct_guess else 0

        # we've seen all the frames
        terminated = self.current_frame_idx == self._num_frames

        observation = self._get_observation(self.current_frame_idx)
        info = self._get_info()

        self._last_action = action
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
