from gymnasium.envs.registration import register

register(
    id='rubiks-cube-222-v0',
    entry_point='rubiks_cube_gym.envs:RubiksCube222Env',
)

register(
    id='rubiks-cube-222-dqn-v0',
    entry_point='rubiks_cube_gym.envs:RubiksCube222DQNEnv',
    max_episode_steps=50,
)

register(
    id='rubiks-cube-222-one-face-v0',
    entry_point='rubiks_cube_gym.envs:RubiksCube222EnvOneFace',
    max_episode_steps=250,
)
