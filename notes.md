# Miscellaneous notes

- "computational modeling of gaze dynamics as exhibited by humans when perceiving socially relevant multimodal information" => model-based RL?

- inverse RL for analysing gaze dynamics (i.e., what reward are humans optimising, when confronted with "socially relevant multimodal information")
  - does it match the proposed model in the paper?
  - if one manages to extract the reward from the trajectories, what about doing classic RL with that same reward? maybe there are more efficient way of maximising it!

- salience vs gaze dynamics! => key is not selection (i.e., space), but time ("when does a forager leave one patch for the next one?")

- "Then, if the ultimate objective of an active perceiver is total reward maximization, reward can be related to the “internal” value" => liking + wanting

- "The basic foraging dimensions of value-based patch selection and patch handling over time pave the way for analysing in a principled framework social gaze as related to persons’ intentions, feelings, traits, and expertise by exploiting semantically rich multimodal dynamic scenes."

- saccade and fixate := large relocatios followed by local clusterings of gaze points

- "What defines valuable a patch? How is gaze guided within and between patches?"

- Ornstein-Uhlenbeck process operating at different scales => would it be easier to have two RL agents instead of one? could possibly make things more interpretable?
  - in general, exploration/exploitation trade-off seems like a bit of an abuse of terminology => what happens looks to me more akin to exploration at different scales (exploitation comes, probably, later, e.g., when one acts based on the information obtained from the stimulus); i suppose exploration/exploitation makes perfect sense from the foraging side
  - intrinsic vs extrinsic reward!

- saccadic models are available online (check the "on gaze deployment..." paper, section VI-A)

- "The fact that the models obtained after the ablation of high level information (speaker/no-speaker, face location) produce significantly lower scores, highlights the causal effect of the presence of (talking) faces, or more generally top down cues, on attention allocation. This fact has been previously demonstrated in the psychological field..."

- "Expression perception is one fundamental mean for our understanding of and engagement in social interactions. This aspect is intimately related to the notion of value proposed in our work, which represents as a matter of fact a doorway to intertwine attention, cognition and emotion."
