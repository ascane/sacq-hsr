"""
    An implementation of SAC-Q, written on top of the SAC-U implementation
"""

from collections import defaultdict

from policy import BoltzmannPolicy
from q_network import QNetwork
from q_table import QTable
from sacx.sacu import SACU
from sacx.tasked_p_network import PolicyNetwork


class SACQ(SACU):
    def __init__(self, env, qmodel, amodel, tasks, gamma=None, num_learn=10,
                 steps_per_episode=1000, scheduler_period=150, num_avg_gradient=10,
                 listeners=None, temperature=1):
        if gamma is None:
            gamma = qmodel.gamma
        super(SACQ, self).__init__(env, qmodel, amodel, tasks, gamma, num_learn, steps_per_episode, scheduler_period,
                         num_avg_gradient, listeners)
        self.Q = QTable()
        self.M = defaultdict(lambda: 0)
        self.scheduler = self.Q.derive_policy(BoltzmannPolicy, lambda x: self.tasks, temperature=temperature)

    def train_scheduler(self, tau, Tau):
        main_task = self.tasks[0]
        xi = self.scheduler_period
        main_rewards = [r[main_task] for _, _, r, _ in tau]
        for h in range(len(Tau)):
            R = sum([r * self.gamma**k for k, r in enumerate(main_rewards[h*xi:])])
            self.M[Tau[h]] += 1
            # self.Q[tuple(Tau[:h]), Tau[h]] += (R - self.Q[tuple(Tau[:h]), Tau[h]])/self.M[Tau[h]]

            # We used a Q-Table with 0.1 learning rate to update the values in the table.
            # Change 0.1 to the desired learning rate
            self.Q[tuple(Tau[:h]), Tau[h]] += 0.1 * (R - self.Q[tuple(Tau[:h]), Tau[h]])

    def schedule_task(self, Tau):
        return self.scheduler.sample(tuple(Tau))
