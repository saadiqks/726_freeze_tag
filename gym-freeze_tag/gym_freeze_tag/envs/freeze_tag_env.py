import gym
import pyglet
import random
import numpy as np
from pyglet.gl import *
from gym.utils import seeding
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.envs.classic_control import rendering

TAGGERS = 2
AGENTS = 4
MOVEMENT = 3
SCREEN_DIM = 1000
TIME_LIMIT = 1200 # 20 seconds

# TODO: write freezing logic
class FreezeTagEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # (0, 1, 2, 3) = (UP, RIGHT, DOWN, LEFT)

        self.action_space = spaces.Discrete(4)

       # agent image ∈ [0, 255] × [0, 1000] × [0, 1000]
       # agent observation is (self image, allies image, enemies image)

        low = np.array([[(0, 0, 0)] * 3] * (TAGGERS + AGENTS))
        high = np.array([[(255, 1000, 1000)] * 3] * (TAGGERS + AGENTS))

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.observation = None

        self.taggers = [None] * TAGGERS
        self.free_agents = [None] * AGENTS
        self.tagger_trans = [None] * TAGGERS
        self.free_agent_trans = [None] * AGENTS

        self.radius = 50 # radius of every agent and tagger

        self.frozen_agents = []
        radius = self.radius

        # Render agents and tagger
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_DIM, SCREEN_DIM)

            for i in range(TAGGERS):
                self.taggers[i] = rendering.make_circle(radius)
                self.taggers[i].set_color(0, 0, 0) # Tagger is black
                self.viewer.add_geom(self.taggers[i])

                self.tagger_trans[i] = rendering.Transform()
                self.taggers[i].add_attr(self.tagger_trans[i])

            for i in range(AGENTS):
                self.free_agents[i] = rendering.make_circle(radius)
                self.free_agents[i].set_color(0.8, 0.1, 0.1) # Free agent is red
                self.viewer.add_geom(self.free_agents[i])

                self.free_agent_trans[i] = rendering.Transform()
                self.free_agents[i].add_attr(self.free_agent_trans[i])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Prey rewards come before predator rewards
        # Compute aggregate rewards for each category
        reward = [random.random(), random.random()]

        # Calculating reward for taggers
#        for i in range(TAGGERS):
#            reward[i] = len(self.frozen_agents)

        # Calculating reward for agents
#        for i in range(TAGGERS, TAGGERS + AGENTS):
#            reward[i] = AGENTS - len(self.frozen_agents)

        state = self.state

        # Update x and y values for agents and taggers based on action
        for i in range(0, len(state), 2):
            j = i // 2

            a = action[j]

            if a == 0:
                state[i + 1] += MOVEMENT # y coord increases 
            elif a == 1:
                state[i] += MOVEMENT # x coord increases 
            elif a == 2:
                state[i + 1] -= MOVEMENT # y coord decreases 
            elif a == 3:
                state[i] -= MOVEMENT # x coord decreases 

        done = False

        observation = []

        for tagger in self.taggers:
            tagger_self_im = self.get_images(self.viewer, tagger, "tagger", "self")
            tagger_allies_im = self.get_images(self.viewer, tagger, "tagger", "allies")
            tagger_enems_im = self.get_images(self.viewer, tagger, "tagger", "enems")

            gs_self = np.mean(tagger_self_im, -1)
            gs_allies = np.mean(tagger_allies_im, -1)
            gs_enems = np.mean(tagger_enems_im, -1)

            observation.append(np.dstack((gs_self, gs_allies, gs_enems)))

        for free_agent in self.free_agents:
            free_agent_self_im = self.get_images(self.viewer, free_agent, "free_agent", "self")
            free_agent_allies_im = self.get_images(self.viewer, free_agent, "free_agent", "allies")
            free_agent_enems_im = self.get_images(self.viewer, free_agent, "free_agent", "enems")

            gs_self = np.mean(free_agent_self_im, -1)
            gs_allies = np.mean(free_agent_allies_im, -1)
            gs_enems = np.mean(free_agent_enems_im, -1)

            observation.append(np.dstack((gs_self, gs_allies, gs_enems)))

        self.observation = observation
        # observation is a list of 3-tuples, where each 3-tuple contains a self, allies, and enemies image 
        return observation, reward, done, {}

    # For a given geom, returns self, allies or enemies image
    def get_images(self, viewer, geom, category, mode):
        glClearColor(1, 1, 1, 1)
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
        observation = []

        for tagger in self.taggers:
            tagger_self_im = self.get_images(self.viewer, tagger, "tagger", "self")
            tagger_allies_im = self.get_images(self.viewer, tagger, "tagger", "allies")
            tagger_enems_im = self.get_images(self.viewer, tagger, "tagger", "enems")

            gs_self = np.mean(tagger_self_im, -1)
            gs_allies = np.mean(tagger_allies_im, -1)
            gs_enems = np.mean(tagger_enems_im, -1)

            observation.append(np.dstack((gs_self, gs_allies, gs_enems)))

        for free_agent in self.free_agents:
            free_agent_self_im = self.get_images(self.viewer, free_agent, "free_agent", "self")
            free_agent_allies_im = self.get_images(self.viewer, free_agent, "free_agent", "allies")
            free_agent_enems_im = self.get_images(self.viewer, free_agent, "free_agent", "enems")

            gs_self = np.mean(free_agent_self_im, -1)
            gs_allies = np.mean(free_agent_allies_im, -1)
            gs_enems = np.mean(free_agent_enems_im, -1)

            observation.append(np.dstack((gs_self, gs_allies, gs_enems)))

        self.observation = observation

        return self.observation

    def render(self, mode='human'):

        if self.state is None:
            return None

        for i in range(TAGGERS):
            self.tagger_trans[i].set_translation(self.state[2 * i], self.state[2 * i + 1])

        for i in range(TAGGERS, TAGGERS + AGENTS):
            self.free_agent_trans[i - TAGGERS].set_translation(self.state[2 * i], self.state[2 * i + 1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
