from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from dqn import DQNAgent
from policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from multi_core import MultiProcessor, MultiAgentFramework
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

# modified to use multiprocessor - processes multi agent step correctly
class AtariProcessor(MultiProcessor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation.astype('uint8'))
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
# add freeze tag env name as default arg
parser.add_argument('--env-name', type=str, default='gym_freeze_tag:freeze_tag-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our hider_model. We use the same hider_model that was described by Mnih et al. (2015).
hider_model = Sequential()
hider_model.add(Flatten(input_shape=(1,) + (18,)))
# hider_model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
# hider_model.add(Activation('relu'))
# hider_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
# hider_model.add(Activation('relu'))
# hider_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
# hider_model.add(Activation('relu'))
# hider_model.add(Flatten())
hider_model.add(Dense(512))
hider_model.add(Activation('relu'))
hider_model.add(Dense(512))
hider_model.add(Activation('relu'))
hider_model.add(Dense(512))
hider_model.add(Activation('relu'))
hider_model.add(Dense(nb_actions))
hider_model.add(Activation('linear'))
print(hider_model.summary())

seeker_model = Sequential()
seeker_model.add(Flatten(input_shape=(1,) + (18,)))
# seeker_model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
# seeker_model.add(Activation('relu'))
# seeker_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
# seeker_model.add(Activation('relu'))
# seeker_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
# seeker_model.add(Activation('relu'))
# seeker_model.add(Flatten())
seeker_model.add(Dense(512))
seeker_model.add(Activation('relu'))
seeker_model.add(Dense(512))
seeker_model.add(Activation('relu'))
seeker_model.add(Dense(512))
seeker_model.add(Activation('relu'))
seeker_model.add(Dense(nb_actions))
seeker_model.add(Activation('linear'))
print(seeker_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=1)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=50000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

# creating both agents
hider_dqn = DQNAgent(model=hider_model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=100, gamma=.99, target_model_update=10,
               train_interval=2, delta_clip=1.)
hider_dqn.compile(Adam(lr=.00025), metrics=['mae'])
seeker_dqn = DQNAgent(model=seeker_model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=100, gamma=.99, target_model_update=10,
               train_interval=2, delta_clip=1.)
seeker_dqn.compile(Adam(lr=.00025), metrics=['mae'])

# passing both agents to framework
framework = MultiAgentFramework(dqagents=[hider_dqn,seeker_dqn])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    #callbacks += [FileLogger(log_filename, interval=100)]
    framework.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000,visualize=False)

    # After training is done, we save the final weights one more time.
    framework.save_weights(weights_filename, overwrite=True)

    # Before executing test, copy changes made to fit() to ensure test() works in multi-agent setting
    framework.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    framework.load_weights(weights_filename)
    framework.test(env, nb_episodes=10, visualize=True)
