import random
from experiment_util import Configurable
"""
    Core classes of a Reinforcement Learning experiment
    
    NOTE: Only fully observable environments are supported in this implementation!
"""


class Action(object):
    """
        Action to be performed on an environment
    """
    pass


class State(object):
    """
        Environment state obtained from executing an action in the environment
    """

    def __init__(self, terminal):
        """
        Create a new state
        :param terminal: A boolean that indicates if the environment state is terminal
        """
        self.terminal = terminal

    def is_terminal(self):
        """
        :return: a boolean indicating if the environment state is terminal
        """
        return self.terminal


class Environment(Configurable):
    """
        Class for describing the environments and how they handle states/actions/rewards/observations for the algorithms
        to learn from
    """

    def sample(self):
        """
        Uniformly sample an action that can be performed on the current environment state
        :return: the sampled action
        """
        raise NotImplementedError

    def step(self, action):
        """
        Perform the action on the current model state. Return an observation and a corresponding reward
        :param action: The action to be performed
        :return: A two-tuple of
                        - a state observation
                        - reward obtained from performing the action
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the internal model state
        :return: an initial observation
        """
        raise NotImplementedError


class FiniteActionEnvironment(Environment):
    """
        Class of environments that have a finite set of actions
    """

    @staticmethod
    def valid_actions_from(state):
        """
        Get all valid actions that can be executed on the state
        :param state: The state on which the actions should be executed
        :return: a list of valid actions
        """
        raise NotImplementedError

    @staticmethod
    def action_space():
        """
        Get all actions that can possibly occur when running the environment
        :return: a list of possible actions
        """
        raise NotImplementedError

    def valid_actions(self):
        """
        :return: a list of actions that can be performed on the current environment state
        """
        raise NotImplementedError

    def sample(self):
        """
        Uniformly sample an action from the valid actions
        :return: the sampled action
        """
        actions = self.valid_actions()
        return actions[random.randint(0, len(actions) - 1)]

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
