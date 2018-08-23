from core import State, Action, FiniteActionEnvironment
import gym
import gym_gazebo_hsr  # This registers the env
import math
from sacx.extcore import TaskEnvironment, Task


class GazeboHsrState(State):
    """
        GazeboHsrState
        :param state: A HsrState, a state from GazeboHsrAssemblyEnv, observation after a step.
        :param terminal: Boolean.
    """

    def __init__(self, state, terminal):
        super(GazeboHsrState, self).__init__(terminal)
        self.state = state

    def __str__(self):
        return str(self.state)


class GazeboHsrAction(Action):
    """
        GazeboHsr Environment Action
    """

    def __init__(self, value):
        """
        Create a new GazeboHsr Action
        :param value: A int indicating which action to take
        """
        self.value = value

MAIN_TASK = Task("main")
REACH = Task("Reach")
MOVE = Task("Move")
LIFT = Task("Lift")

TASKS =[MAIN_TASK, REACH, MOVE, LIFT]


class GazeboHsr(FiniteActionEnvironment, TaskEnvironment):
    """
        MountainCar environment class
    """

    UP = GazeboHsrAction(0)
    DOWN = GazeboHsrAction(1)
    RIGHT = GazeboHsrAction(2)
    LEFT = GazeboHsrAction(3)
    FORWARD = GazeboHsrAction(4)
    BACKWARD = GazeboHsrAction(5)
    OPEN = GazeboHsrAction(12)
    CLOSE = GazeboHsrAction(13)
    ACTIONS = [UP, DOWN, FORWARD, BACKWARD, OPEN, CLOSE]
    # ACTIONS = [UP, DOWN, RIGHT, LEFT, FORWARD, BACKWARD, OPEN, CLOSE]

    def __init__(self, render=True):
        """
        Create a new GazeboHsrEnvironment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super(GazeboHsr, self).__init__()
        self.env = gym.make('gazebo-hsr-assembly-v0')
        self.render = render

        self.terminal = False
        self.step_v = 0

        self.reset()

    @staticmethod
    def action_space():
        return list(GazeboHsr.ACTIONS)

    @staticmethod
    def valid_actions_from(state):
        # TODO: Some actions are not valid
        return GazeboHsr.action_space()

    def valid_actions(self):
        # TODO: Some actions are not valid
        return GazeboHsr.action_space()

    @staticmethod
    def norm2(v1, v2):
        return math.sqrt(math.pow(v1.x - v2.x, 2) + math.pow(v1.y - v2.y, 2) + math.pow(v1.z - v2.z, 2))

    def reach_reward(self):
        box_green_position = self.env.get_model_pose('box_green').position
        hand_position = self.env.get_hand_position()
        dist = GazeboHsr.norm2(box_green_position, hand_position)
        return 1 - dist if dist < 1 else 0

    def move_reward(self, prev_box_green_position):
        box_green_position = self.env.get_model_pose('box_green').position
        # We only care about horizontal move
        box_green_position.z = prev_box_green_position.z
        dist = GazeboHsr.norm2(prev_box_green_position, box_green_position)
        return 10 * dist

    def lift_reward(self):
        box_green_position = self.env.get_model_pose('box_green').position
        return box_green_position.z - 0.565 if box_green_position.z > 0.565 else 0

    def step(self, action):
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        prev_box_green_position = self.env.get_model_pose('box_green').position
        obs, reward, self.terminal, info = self.env.step(action.value)
        self.step_v += 1
        rewards = {
            MAIN_TASK: reward,
            REACH: self.reach_reward(),
            MOVE: self.move_reward(prev_box_green_position),
            LIFT: self.lift_reward()
        }
        print(reward)

        return GazeboHsrState(obs, self.terminal), rewards

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.terminal = False
        self.step_v = 0
        return GazeboHsrState(self.env.reset(), self.terminal)

    @staticmethod
    def auxiliary_tasks():
        return TASKS[1:]

    @staticmethod
    def get_tasks():
        return TASKS


if __name__ == '__main__':

    _e = GazeboHsr(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
