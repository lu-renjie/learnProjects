import gym
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter

from reinforce import REINFORCE

ex = Experiment()
ex.add_config('config.yaml')

@ex.config
def config(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert len(state_dim) == 1, f'{state_dim}'

    state_dim = state_dim[0]
    print('state_dim: ', state_dim)
    print('action_dim: ', action_dim)
    del env


def test(env_name, agent):
    N = 3
    G = 0
    env = gym.make(env_name)
    for _ in range(N):  # run N times for more accurate test
        s, _ = env.reset()
        while True:
            a = agent.take_action(s, deterministic=True)
            s, r, terminated, truncated, _ = env.step(a)
            G += r
            if terminated or truncated:
                break
    return G / N


@ex.automain
def main(_config):
    writer = SummaryWriter(comment=_config['env_name'])
    writer.add_text(
        'hyperparameters',
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in _config.items()])
    )

    env = gym.make(_config['env_name'])
    agent = REINFORCE(_config)

    s, _ = env.reset()
    for step in range(_config['max_step']):
        a = agent.take_action(s)
        s_, r, terminated, truncated, _ = env.step(a)
        agent.store(s, a, r)
        s = s_

        if step % _config['test_interval'] == 0:
            G = test(_config['env_name'], agent)
            writer.add_scalar('return', G, step)
            print(f'step: {step:06d}, return:{G:8.4f}')
        if terminated or truncated:
            s, _ = env.reset()
            agent.learn()   # 每个episode更新一次
    env.close()
    writer.close()
