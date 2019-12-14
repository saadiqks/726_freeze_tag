import gym
import pyglet
import random
import numpy as np
#from pyglet.gl import *
from gym.utils import seeding
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.envs.classic_control import rendering

PREDS = 2
PREY = 4
MOVEMENT = 3
SCREEN_DIM = 1000
STEPS_LIMIT = 2000

class FreezeTagEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # (0, 1, 2, 3) = (UP, RIGHT, DOWN, LEFT)

        self.action_space = spaces.Discrete(4)

       # agent image ∈ [0, 255] × [0, 1000] × [0, 1000]
       # agent observation is (self image, allies image, enemies image)

        low = np.array([[(0, 0, 0)] * 3] * (PREDS + PREY))
        high = np.array([[(255, 1000, 1000)] * 3] * (PREDS + PREY))

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.observation = None
        self.counter = 0

        self.predators = [None] * PREDS
        self.free_agents = [None] * PREY
        self.tagger_trans = [None] * PREDS
        self.free_agent_trans = [None] * PREY
        self.frozen_agents = [0] * PREY

        self.radius = 50 # radius of every agent and tagger

        radius = self.radius

        self.penalty = -1000

        # Render agents and tagger
        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_DIM, SCREEN_DIM)

            for i in range(PREDS):
                self.predators[i] = rendering.make_circle(radius)
                self.predators[i].set_color(0, 0, 0) # Predator is black
                self.viewer.add_geom(self.predators[i])

                self.tagger_trans[i] = rendering.Transform()
                self.predators[i].add_attr(self.tagger_trans[i])

            for i in range(PREY):
                self.free_agents[i] = rendering.make_circle(radius)
                self.free_agents[i].set_color(0.8, 0.1, 0.1) # Prey is red
                self.viewer.add_geom(self.free_agents[i])

                self.free_agent_trans[i] = rendering.Transform()
                self.free_agents[i].add_attr(self.free_agent_trans[i])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def collision(self, x_1, y_1, x_2, y_2):
        z_1 = np.array([x_1, y_1])
        z_2 = np.array([x_2, y_2])

        return np.linalg.norm(z_1 - z_2) <= 2 * self.radius

    def step(self, action):
        # Prey rewards come before predator rewards
        # Compute aggregate rewards for each category
        reward = [0, 0]

        pred_out_of_bounds = 0
        prey_out_of_bounds = 0
        state = self.state

        # Calculating reward for predators, penalizing them if they leave screen
        for i in range(PREDS):
            # Check if tagger froze any free agent and that the agent is currently unfrozen
            for j in range(PREY):
                if self.collision(state[2 * i], state[2 * i + 1], state[2 * (PREDS + j)], state[2 * (PREDS + j) + 1]) and not self.frozen_agents[j]:
                    self.frozen_agents[j] = 1
                    reward[1] += 1
                    reward[0] -= 1
            # Penalizing for leaving screen
            if state[2 * i] < 0 or state[2 * i] > 1000 or state[2 * i + 1] < 0 or state[2 * i + 1] > 1000:
                pred_out_of_bounds += 1
                reward[1] = self.penalty * pred_out_of_bounds

        # Calculating reward for agents, penalizing them if they leave screen
        for i in range(PREDS, PREDS + PREY):
            # Check if any free agent unfroze any of its frozen brethren 
            for j in range(PREY):
                if i != PREDS + j:
                    if self.collision(state[2 * i], state[2 * i + 1], state[2 * (PREDS + j)], state[2 * (PREDS + j) + 1]) and self.frozen_agents[j]:
                        self.frozen_agents[j] = 0
                        reward[0] += 1
                        reward[1] -= 1
            if state[2 * i] < 0 or state[2 * i] > 1000 or state[2 * i + 1] < 0 or state[2 * i + 1] > 1000:
                prey_out_of_bounds += 1
                reward[0] = self.penalty * prey_out_of_bounds

        # Update x and y values for agents and predators based on action
        for i in range(0, len(state), 2):
            j = i // 2

            if j < 2 or not self.frozen_agents[j - 2]: # Ensures frozen agents do not move
                a = action[j]

                if a == 0:
                    if state[i + 1] + MOVEMENT < 1000:
                        state[i + 1] += MOVEMENT # y coord increases 
                elif a == 1:
                    if state[i] + MOVEMENT < 1000:
                        state[i] += MOVEMENT # x coord increases 
                elif a == 2:
                    if state[i + 1] - MOVEMENT > 0:
                        state[i + 1] -= MOVEMENT # y coord decreases 
                elif a == 3:
                    if state[i] - MOVEMENT > 0:
                        state[i] -= MOVEMENT # x coord decreases 

        done = False

        observation = []

        for i in range(PREDS):
            self.tagger_trans[i].set_translation(self.state[2 * i], self.state[2 * i + 1])

        for i in range(PREDS, PREDS + PREY):
            self.free_agent_trans[i - PREDS].set_translation(self.state[2 * i], self.state[2 * i + 1])

        prey_1 = [self.state[0], self.state[1], self.frozen_agents[0]]
        prey_2 = [self.state[2], self.state[3], self.frozen_agents[1]]
        prey_3 = [self.state[4], self.state[5], self.frozen_agents[2]]
        prey_4 = [self.state[6], self.state[7], self.frozen_agents[3]]
        pred_1 = [self.state[8], self.state[9], 0]
        pred_2 = [self.state[10], self.state[11], 0]

        # [self, other prey, predators]
        observation.append([prey_1 + prey_2 + prey_3 + prey_4 + pred_1 + pred_2][0])
        observation.append([prey_2 + prey_3 + prey_4 + prey_1 + pred_1 + pred_2][0])
        observation.append([prey_3 + prey_4 + prey_1 + prey_2 + pred_1 + pred_2][0])
        observation.append([prey_4 + prey_1 + prey_2 + prey_3 + pred_1 + pred_2][0])

        # [self, other pred, prey]
        observation.append([pred_1 + pred_2 + prey_1 + prey_2 + prey_3 + prey_4][0])
        observation.append([pred_2 + pred_1 + prey_1 + prey_2 + prey_3 + prey_4][0])

        """
        for tagger in self.predators:
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
        """

        self.observation = observation
        self.state = state

        self.counter += 1

        if self.counter > STEPS_LIMIT:
            done = True

        return self.observation, reward, done, {}

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
                for tagger in self.predators:
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
                for tagger in self.predators:
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
        self.counter = 0
        self.state = self.np_random.uniform(0, SCREEN_DIM, size=(2 * PREY + 2 * PREDS),)
        observation = []

        self.frozen_agents = [0] * PREY

        for i in range(PREDS):
            self.tagger_trans[i].set_translation(self.state[2 * i], self.state[2 * i + 1])

        for i in range(PREDS, PREDS + PREY):
            self.free_agent_trans[i - PREDS].set_translation(self.state[2 * i], self.state[2 * i + 1])

        prey_1 = [self.state[0], self.state[1], self.frozen_agents[0]]
        prey_2 = [self.state[2], self.state[3], self.frozen_agents[1]]
        prey_3 = [self.state[4], self.state[5], self.frozen_agents[2]]
        prey_4 = [self.state[6], self.state[7], self.frozen_agents[3]]
        pred_1 = [self.state[8], self.state[9], 0]
        pred_2 = [self.state[10], self.state[11], 0]

        # [self, other prey, predators]
        observation.append([prey_1 + prey_2 + prey_3 + prey_4 + pred_1 + pred_2][0])
        observation.append([prey_2 + prey_3 + prey_4 + prey_1 + pred_1 + pred_2][0])
        observation.append([prey_3 + prey_4 + prey_1 + prey_2 + pred_1 + pred_2][0])
        observation.append([prey_4 + prey_1 + prey_2 + prey_3 + pred_1 + pred_2][0])

        # [self, other pred, prey]
        observation.append([pred_1 + pred_2 + prey_1 + prey_2 + prey_3 + prey_4][0])
        observation.append([pred_2 + pred_1 + prey_1 + prey_2 + prey_3 + prey_4][0])

        """
        for tagger in self.predators:
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
        """
        self.observation = observation

        return self.observation

    def render(self, mode='human'):
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
