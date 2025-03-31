import gym


def get_env(env_name):
    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Discrete):  # 离散动作空间
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        raise '不支持的环境'
    return env, state_dim, action_dim


def get_vec_env(env_name, num_envs):
    env = gym.vector.make(env_name, num_envs=num_envs)
    return env