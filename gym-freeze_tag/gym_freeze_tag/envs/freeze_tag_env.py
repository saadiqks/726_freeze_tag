import gym
import numpy as np
from gym.utils import seeding
from gym import error, spaces, utils

AGENTS = 1
SCREEN_DIM = 1000
class FreezeTagEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
       self.tagger_action_space = spaces.Discrete(4)
       self.agent_action_spaces = [spaces.Discrete(4) for i in range(AGENTS)]

       # tagger loc ∈ [0, 1000]
       # agent loc ∈ [0, 1000]
       low = np.array([0] * (AGENTS + 1))
       high = np.array([SCREEN_DIM] * (AGENTS + 1))
       self.observation_space = spaces.Box(low, high, dtype=np.float32)

       self.seed()
       self.viewer = None
       self.state = None

       self.radius = 50 # radius of every agent and tagger

       self.frozen_agents = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state

        tagger_x, tagger_y, first_agent_x, first_agent_y = state

        # Update x and y values for agents and tagger based on action

        self.frozen_agents += 1

        reward = None

        tagger_x += 1
        tagger_y += 1

        first_agent_x += 1
        first_agent_y += 1

        self.state = (tagger_x, tagger_y, first_agent_x, first_agent_y)

        done = self.frozen_agents == AGENTS

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(0, SCREEN_DIM, size=(2 * AGENTS + 2),)

        return np.array(self.state)

    def render(self, mode='human'):
        radius = self.radius

        tagger_x, tagger_y, first_agent_x, first_agent_y = self.state

        # Render agents and tagger
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(SCREEN_DIM, SCREEN_DIM)

            tagger = rendering.make_circle(radius)
            first_agent = rendering.make_circle(radius)

            tagger.set_color(0, 0, 0)
            first_agent.set_color(1, 0, 0)

            self.tagger_trans = rendering.Transform()
            self.first_agent_trans = rendering.Transform()

            self.viewer.add_geom(tagger)
            self.viewer.add_geom(first_agent)

            tagger.add_attr(self.tagger_trans)
            first_agent.add_attr(self.first_agent_trans)

        if self.state is None:
            return None

        self.tagger_trans.set_translation(tagger_x, tagger_y)
        self.first_agent_trans.set_translation(first_agent_x, first_agent_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

x = FreezeTagEnv()
