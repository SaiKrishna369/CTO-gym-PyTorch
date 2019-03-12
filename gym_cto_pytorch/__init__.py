from gym.envs.registration import register

register(
    id='TCTO-v0',
    entry_point='gym_cto_pytorch.envs:CtoEnv',
)

register(
    id='TCTO-v1',
    entry_point='gym_cto_pytorch.envs:eCtoEnv',
)