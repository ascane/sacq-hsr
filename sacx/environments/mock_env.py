import numpy as np

from core import FiniteActionEnvironment, State, Action
from sacx.extcore import TaskEnvironment


ACTIONS = [0, 1, 2]

TASKS = [0,1,2]

class MockState(State):
    def __init__(self, terminal, state):
        super(MockState, self).__init__(terminal)
        self.state = state


class MockEnv(FiniteActionEnvironment, TaskEnvironment):
    @staticmethod
    def valid_actions_from(state):
        return ACTIONS

    @staticmethod
    def action_space():
        return ACTIONS

    def valid_actions(self):
        return ACTIONS

    def step(self, action):
        rewards = {i: 1 if action == i else 0 for i in range(3)}

        return MockState(False, np.random.normal(0, 1, (5,))), rewards

    def reset(self):
        return MockState(False, np.random.normal(0, 1, (5,)))

    @staticmethod
    def auxiliary_tasks():
        return TASKS[1:]

    @staticmethod
    def get_tasks():
        return TASKS