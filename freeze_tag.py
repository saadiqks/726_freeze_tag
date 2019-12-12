import gym

env = gym.make('gym_freeze_tag:freeze_tag-v0')
for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
#        print(observation)
        action = [env.action_space.sample(), env.action_space.sample(), env.action_space.sample(), env.action_space.sample(), env.action_space.sample(), env.action_space.sample()]
        observation, reward, done, _ = env.step(action)

        if done:
            print(f"Episode finished after {t + 1} timesteps")
            break
env.close()
