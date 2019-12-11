# -*- coding: utf-8 -*-
import warnings
from copy import deepcopy

import numpy as np
from keras.callbacks import History

from rl.callbacks import (
    CallbackList,
    TestLogger,
    TrainEpisodeLogger,
    TrainIntervalLogger,
    Visualizer,
    WandbLogger
)

class MultiAgentFramework(object):
	def __init__(self, dqagents, processor=None):
        # get DQN agents at construction time - [0] is hider agent, [1] is seeker agent
        self.dqagents = dqagents
		self.processor = processor
        self.training = False
        self.step = 0

    def load_weights(self, filepath):
        self.dqagents[0].load_weights(filepath+"_hider")
        self.dqagents[1].load_weights(filepath+"_seeker")

        self.dqagents[0].update_target_model_hard()
        self.dqagents[1].update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.dqagents[0].save_weights(filepath + "_hider", overwrite=overwrite)
        self.dqagents[1].save_weights(filepath + "_seeker", overwrite=overwrite)
	
	def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
            visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
            nb_max_episode_steps=None):

		for dqagent in self.dqagents:
        	if not dqagent.compiled:
            	raise RuntimeError(
                	'Your tried to fit your agents but one hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        	if action_repetition < 1:
            	raise ValueError(
                	'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = True

        callbacks = [] if not callbacks else callbacks[:]

        if verbose == 1:
            callbacks += [TrainIntervalLogger(interval=log_interval)]
        elif verbose > 1:
            callbacks += [TrainEpisodeLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_steps': nb_steps,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)
        self._on_train_begin()
        callbacks.on_train_begin()

        episode = np.int16(0)
        self.step = np.int16(0)
        observations = []
        episode_reward = None
        episode_step = None
        did_abort = False
        try:
            while self.step < nb_steps:
                # check if observations is empty
                if observations == []:  # start of a new episode
                    callbacks.on_episode_begin(episode)
                    episode_step = np.int16(0)
                    episode_reward = np.float32(0)

                    # Obtain the initial observation by resetting the environment.
                    self.reset_states()
                    observations = deepcopy(env.reset())
                    if self.processor is not None:
                        # process all observations
                        observations = [self.processor.process_observation(
                            observation) for observation in observations]
                    assert observations != []

                    # Perform random starts at beginning of episode and do not record them into the experience.
                    # This slightly changes the start position between games.

                    # can remove this bit, not gonna use any random starts
                    nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                        nb_max_start_steps)
                    for _ in range(nb_random_start_steps):
                        if start_step_policy is None:
                            actions = env.action_space.sample()
                        else:
                            actions = start_step_policy(observation)
                        if self.processor is not None:
                            actions = self.processor.process_action(action)
                        callbacks.on_action_begin(action)
                        observations, rewards, done, info = env.step(action)
                        observations = deepcopy(observations)
                        if self.processor is not None:
                            observations, rewards, done, info = self.processor.process_step(
                                observations, rewards, done, info)
                        callbacks.on_action_end(action)
                        if done:
                            warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                                nb_random_start_steps))
                            observations = deepcopy(env.reset())
                            if self.processor is not None:
                                observations = [self.processor.process_observation(
                                    observation) for observation in observations]
                            break

                # At this point, we expect to be fully initialized.
                assert episode_reward is not None
                assert episode_step is not None
                assert observations != []

                # Run a single step.
                callbacks.on_step_begin(episode_step)
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).

                # given incides [0,3] are hider indices and [4,5] are seeker indices
                actions = []
                for i in range(0,4):
                    actions.append(self.dqagents[0].forward(observations[i]))
                for i in range(4,6):
                    actions.append(self.dqagents[1].forward(observations[i]))

                # process all actions
                if self.processor is not None:
                    actions = [self.processor.process_action(action) for action in actions]
                reward = np.float32(0)
                accumulated_info = {}
                done = False
                for _ in range(action_repetition):
                    callbacks.on_action_begin(actions)
                    # expect rs[0] to be aggregate hider reward, rs[1] aggregate seeker reward
                    observations, rs, done, info = env.step(actions)
                    observations = deepcopy(observations)
                    if self.processor is not None:
                        observations, rs, done, info = self.processor.process_step(
                            observations, rs, done, info)
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    callbacks.on_action_end(actions)
                    hider_reward += rs[0]
                    seeker_reward += rs[1]
                    if done:
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    # Force a terminal state.
                    done = True

                # run backwrd step wrt each agent's respective aggregate reward
                hider_metrics = self.dqagents[0].backward(hider_reward, terminal=done)
                seeker_metrics = self.dqagents[1].backward(seeker_reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': actions,
                    'observation': observations,
                    'hider_reward': hider_reward,
                    'seeker_reward': seeker_reward,
                    'seeker_metrics': seeker_metrics,
                    'hider_metrics': hider_metrics,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

                if done:
                    # We are in a terminal state but the agent hasn't yet seen it. We therefore
                    # perform one more forward-backward call and simply ignore the action before
                    # resetting the environment. We need to pass in `terminal=False` here since
                    # the *next* state, that is the state of the newly reset environment, is
                    # always non-terminal by convention.
                    for i in range(0,4):
                        self.dqagents[0].forward(observations[i])
                    for i in range(4,6):
                        self.dqagents[1].forward(observations[i])

                    self.dqagents[0].backward(0., terminal=False)
                    self.dqagents[1].backward(0., terminal=False)

                    # This episode is finished, report and reset.
                    episode_logs = {
                        'episode_reward': episode_reward,
                        'nb_episode_steps': episode_step,
                        'nb_steps': self.step,
                    }
                    callbacks.on_episode_end(episode, episode_logs)

                    episode += 1
                    observations = []
                    episode_step = None
                    episode_reward = None
        except KeyboardInterrupt:
            # We catch keyboard interrupts here so that training can be be safely aborted.
            # This is so common that we've built this right into this function, which ensures that
            # the `on_train_end` method is properly called.
            did_abort = True
        callbacks.on_train_end(logs={'did_abort': did_abort})
        self._on_train_end()

        return history

    def test(self, env, nb_episodes=1, action_repetition=1, callbacks=None, visualize=True,
             nb_max_episode_steps=None, nb_max_start_steps=0, start_step_policy=None, verbose=1):
        """Callback that is called before training begins.

        # Arguments
            env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
            nb_episodes (integer): Number of episodes to perform.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.

        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """
		for dqagent in self.dqagents:
        	if not dqagent.compiled:
            	raise RuntimeError(
                	'Your tried to fit your agents but one hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        	if action_repetition < 1:
            	raise ValueError(
                	'action_repetition must be >= 1, is {}'.format(action_repetition))

        self.training = False
        self.step = 0

        callbacks = [] if not callbacks else callbacks[:]

        if verbose >= 1:
            callbacks += [TestLogger()]
        if visualize:
            callbacks += [Visualizer()]
        history = History()
        callbacks += [history]
        callbacks = CallbackList(callbacks)
        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self)
        else:
            callbacks._set_model(self)
        callbacks._set_env(env)
        params = {
            'nb_episodes': nb_episodes,
        }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        self._on_test_begin()
        callbacks.on_train_begin()
        for episode in range(nb_episodes):
            callbacks.on_episode_begin(episode)
            episode_reward = 0
            episode_step = 0

            # Obtain the initial observation by resetting the environment.
            self.reset_states()
            observation = deepcopy(env.reset())
            if self.processor is not None:
                observation = self.processor.process_observation(observation)
            assert observation is not None

            # Perform random starts at beginning of episode and do not record them into the experience.
            # This slightly changes the start position between games.
            nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(
                nb_max_start_steps)
            for _ in range(nb_random_start_steps):
                if start_step_policy is None:
                    action = env.action_space.sample()
                else:
                    action = start_step_policy(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                if self.processor is not None:
                    observation, r, done, info = self.processor.process_step(
                        observation, r, done, info)
                callbacks.on_action_end(action)
                if done:
                    warnings.warn('Env ended before {} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.'.format(
                        nb_random_start_steps))
                    observation = deepcopy(env.reset())
                    if self.processor is not None:
                        observation = self.processor.process_observation(
                            observation)
                    break

            # Run the episode until we're done.
            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                if self.processor is not None:
                    action = self.processor.process_action(action)
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    if self.processor is not None:
                        observation, r, d, info = self.processor.process_step(
                            observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            self.forward(observation)
            self.backward(0., terminal=False)

            # Report end of episode.
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()
        self._on_test_end()

        return history

    def _on_train_begin(self):
        """Callback that is called before training begins."
        """
        pass

    def _on_train_end(self):
        """Callback that is called after training ends."
        """
        pass

    def _on_test_begin(self):
        """Callback that is called before testing begins."
        """
        pass

    def _on_test_end(self):
        """Callback that is called after testing ends."
        """
        pass

class MultiProcessor(object):
    """Abstract base class for implementing processors.

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observations, rewards, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observations = [self.process_observation(observation) for observation in observations]
        rewards = [self.process_reward(reward) for reward in rewards]
        info = self.process_info(info)
        return observations, rewards, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            observation (object): An observation as obtained by the environment

        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            info (dict): An info as obtained by the environment

        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

        # Arguments
            action (int): Action given to the environment

        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.

        # Arguments
            batch (list): List of states

        # Returns
            Processed list of states
        """
        return batch

    @property
    def metrics(self):
        """The metrics of the processor, which will be reported during training.

        # Returns
            List of `lambda y_true, y_pred: metric` functions.
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        """
        return []