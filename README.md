#  gadro

[GazeDeploy](https://github.com/phuselab/GazeDeploy) (more or less), but using Deep Reinforcement Learning.

## Running the code

The code uses `python 3.11.8`. To run it, do the following:

1. Get the data from [PHuSe Lab](https://phuselab.di.unimi.it/)'s SFTP server.
    1. (optional) Spin up a virtual environment.
2. Install all the requirements (i.e, `pip install -r requirements.txt`).
3. Run `utils_extract_frames.py`.
    - This **must** be done if you want to use any of the notebook with `frames` in their name.
4. Pick the Jupyter notebook that most interests you and have fun!
    - Feel free to edit the notebooks at will: I've kept the code in notebook format precisely for thar purpose.

## Code organisation

I have tried to make the function of each file explicit by naming them appropriately.
I have also tried to keep related files close together when sorted, for ease of perusal.

Below, I shall provide a more in-depth look at the organisation:

- all the `env_` files contain environment code, with

  - `base` denoting an environment that only uses numerical features (patch centres, bounding boxes, et alia),
  - `frames` denoting and environment that uses the frame data (i.e., original frames, attention maps and patch bitmaps),
  - `test` denoting environments that are meant to be used for testing (the difference is that testing environments feed the frames to the agent in chronological order, while the training environments do not);

- all the `utils_` file contain utilities, i.e., support code;
- the `policy_gail_prime.py` file contains a customised version of the default `GAILPolicy` provided by Tianshou;
- all the `note_` files are notebooks and contain the `main' code, i.e., the one that it's most fun to play around and the one that pulls together all of the other files.

As for the code itself, it is largely undocumented (although there are a few comments here and there for clarifying the non-obvious bits), but that is because I've tried to keep it as clean as possible, i.e., its function and use should be clear enough from reading it.
