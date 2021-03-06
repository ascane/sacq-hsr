import core
from sacx.multi_task_logger import PlottingMultiTaskLogger

state_size = 131345

if __name__ == '__main__':
    import keras as ks
    from sacx.environments.gazebohsr import GazeboHsr
    from sacx.tasked_q_network import QNetwork
    from sacx.tasked_p_network import PolicyNetwork
    from sacx.sacq import SACQ

    env = GazeboHsr(render=False)

    print(isinstance(env, core.FiniteActionEnvironment))
    tasks = env.get_tasks()
    actions = env.valid_actions()
    print(actions)

    def common_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        return x

    def task_q_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(3, activation='linear')(x)
        return x

    def task_p_net(x):
        x = ks.layers.Dense(100, activation='relu')(x)
        x = ks.layers.Dense(3, activation='softmax')(x)
        return x


    listeners = [PlottingMultiTaskLogger(tasks, 200, ['red', 'blue', 'blue', 'green'])]
    # In state_transformer, s is of type GazeboHsrState.
    q_network = QNetwork((state_size,), actions, tasks, common_net, task_q_net, lambda s: s.state.to_array(),
                         gamma=0.993, alpha=0.001, reward_scale=0.03, fixed_steps=100, lambd=0.95, lambd_min=1e-2)
    p_network = PolicyNetwork((state_size,), actions, tasks, common_net, task_p_net, lambda s: s.state.to_array(),
                              entropy_regularization=0.005, alpha=0.0001, fixed_steps=100)

    agent = SACQ(env, q_network, p_network, tasks, num_learn=30, scheduler_period=200, listeners=listeners,
                 temperature=0.01)

    agent.learn()
