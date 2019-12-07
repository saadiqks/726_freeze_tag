import gym
import pyglet
import numpy as np
from pyglet.gl import *
from gym.utils import seeding
import matplotlib.pyplot as plt
from gym import error, spaces, utils

TAGGERS = 1
AGENTS = 1
SCREEN_DIM = 1000
TIME_LIMIT = 1200 # 20 seconds

class FreezeTagEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # (0, 1, 2, 3) = (UP, RIGHT, DOWN, LEFT)
        low = np.array([0] * (TAGGERS + AGENTS))
        high = np.array([4] * (TAGGERS + AGENTS))

        self.action_space = spaces.Box(low, high, dtype=np.int32)
       # agent image ∈ [0, 255] × [0, 1000] × [0, 1000]
       # agent observation is (self image, allies image, enemies image)

        low = np.array([[(0, 0, 0)] * 3] * (TAGGERS + AGENTS))
        high = np.array([[(255, 1000, 1000)] * 3] * (TAGGERS + AGENTS))

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.taggers = [None]
        self.free_agents = [None]

        self.radius = 50 # radius of every agent and tagger

        self.frozen_agents = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state = self.state

        tagger_x, tagger_y, first_agent_x, first_agent_y = state

        # Tagger rewards come before agent rewards
        reward = [0] * (TAGGERS + AGENTS)

        # Calculating reward for taggers
        for i in range(TAGGERS):
            reward[i] = len(self.frozen_agents)

        # Calculating reward for agents
        for i in range(TAGGERS, TAGGERS + AGENTS):
            reward[i] = AGENTS - len(self.frozen_agents)

        # Update x and y values for agents and tagger based on action
        tagger_x += 1
        tagger_y += 1

        first_agent_x += 1
        first_agent_y += 1

        self.state = (tagger_x, tagger_y, first_agent_x, first_agent_y)

        done = False

        observation = []

        for tagger in self.taggers:
            tagger_self_im = self.get_images(self.viewer, tagger, "tagger", "self")
            tagger_allies_im = self.get_images(self.viewer, tagger, "tagger", "allies")
            tagger_enems_im = self.get_images(self.viewer, tagger, "tagger", "enems")

            observation.append((tagger_self_im, tagger_allies_im, tagger_enems_im))

        for free_agent in self.free_agents:
            free_agent_self_im = self.get_images(self.viewer, free_agent, "free_agent", "self")
            free_agent_allies_im = self.get_images(self.viewer, free_agent, "free_agent", "allies")
            free_agent_enems_im = self.get_images(self.viewer, free_agent, "free_agent", "enems")

            observation.append((free_agent_self_im, free_agent_allies_im, free_agent_enems_im))

#        plt.imsave(f"TAGGER_{int(tagger_x)}.png", observation[0][0])
#        plt.imsave(f"FA_{int(first_agent_x)}.png", observation[1][0])

        # observation is a list of 3-tuples, where each 3-tuple contains a self, allies, and enemies image 
        return observation, reward, done, {}

    # For a given geom, returns self, allies or enemies image
    def get_images(self, viewer, geom, category, mode):
        glClearColor(1,1,1,1)
        viewer.window.clear()
        viewer.window.switch_to()
        viewer.window.dispatch_events()
        viewer.transform.enable()

        if category == "tagger":
            if mode == "self":
                geom.render()
            elif mode == "allies":
                for tagger in self.taggers:
                    if tagger != geom:
                        tagger.render()
            elif mode == "enems":
                for fa in self.free_agents:
                    fa.render()

        if category == "free_agent":
            if mode == "self":
                geom.render()
            elif mode == "allies":
                for fa in self.free_agents:
                    if fa != geom:
                        fa.render()
            elif mode == "enems":
                for tagger in self.taggers:
                    tagger.render()

        viewer.transform.disable()
        arr = None
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.data, dtype=np.uint8)
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1,:,0:3]
        viewer.window.flip()

        return arr

    def reset(self):
        self.state = self.np_random.uniform(0, SCREEN_DIM, size=(2 * AGENTS + 2 * TAGGERS),)

        return np.array(self.state)

    def render(self, mode='human'):
        radius = self.radius

        tagger_x, tagger_y, first_agent_x, first_agent_y = self.state

        # Render agents and tagger
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(SCREEN_DIM, SCREEN_DIM)

            self.taggers[0] = rendering.make_circle(radius)
            self.free_agents[0] = rendering.make_circle(radius)

            self.taggers[0].set_color(0, 0, 0) # Tagger is black
            self.free_agents[0].set_color(1, 0, 0) # Free agent is red

            self.tagger_trans = rendering.Transform()
            self.free_agent_trans = rendering.Transform()

            self.viewer.add_geom(self.taggers[0])
            self.viewer.add_geom(self.free_agents[0])

            self.taggers[0].add_attr(self.tagger_trans)
            self.free_agents[0].add_attr(self.free_agent_trans)

        if self.state is None:
            return None

        self.tagger_trans.set_translation(tagger_x, tagger_y)
        self.free_agent_trans.set_translation(first_agent_x, first_agent_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
