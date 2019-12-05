from gym.envs.registration import register

register(
    id='freeze_tag-v0',
    entry_point='gym_freeze_tag.envs:FreezeTagEnv',
)
