import sacx.extcore as ecore
import core


class MultiTaskWrapper(ecore.TaskEnvironment, core.FiniteActionEnvironment):
    def __init__(self, env, reward_fn, tasks):
        """
        Initializes the multi task wrapper
        :param env: The normal environment
        :param reward_fn: A function that generates a reward dict from these parameters: s, a, r, tasks
        """
        super(MultiTaskWrapper, self).__init__()
        self.reward_fn = reward_fn
        self.env = env
        self.tasks = tasks

    def step(self, action):
        s, r = self.env.step(action)
        return s, self.reward_fn(s, action, r, self.tasks)

    def reset(self):
        return self.env.reset()

    def sample(self):
        return self.env.sample()

    def get_tasks(self):
        return self.tasks

    def valid_actions(self):
        return self.env.valid_actions()

