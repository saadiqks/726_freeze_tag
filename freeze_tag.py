import gym

env = gym.make('gym_freeze_tag:freeze_tag-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(5):
        env.render()
        #print(observation)
        action = [env.action_space.sample()] * 6
        observation, reward, done, _ = env.step(action)

        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
env.close()
