import os.path as osp
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ppo_discrete import PPO
from env import get_env, get_vec_env
from logger import Logger


def dict_to_markdown_table(dict):
    table_head = '|param|value|\n|-|-|\n'
    table_content = "\n".join([f"|{key}|{value}|" for key, value in dict.items()])
    return table_head + table_content


class Program:
    def __init__(self, cfg):
        # 把字典数据改成类的成员变量
        for key, value in cfg.items():
            setattr(self, key, value)
    
        # env and agent
        self.env, state_dim, action_dim = get_env(self.env_name)
        self.envs = get_vec_env(self.env_name, self.num_envs)
        cfg['state_dim'] = state_dim
        cfg['action_dim'] = action_dim
        self.agent = PPO(cfg)

        # logger 设置
        if self.log:
            time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            dir1 = f'MLP{self.hidden_num}_{self.hidden_dim}_{self.env_name}'
            dir2 = f'{time}_{self.description}'
            self.log_dir = osp.join('log', dir1, dir2)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.writer.add_text('hyperparameters', dict_to_markdown_table(cfg))
            self.logger = Logger(log_dir=self.log_dir)
        else:
            self.logger = Logger()

    def train(self):
        step = 0
        update_num = self.max_step // (self.num_envs * self.sample_num)

        s, _ = self.envs.reset()
        for update in range(update_num):
            # rollout
            self.agent.buffer.reset()
            for _ in range(self.sample_num):
                a, probs = self.agent.take_action_train(s)
                s_, r, terminated, _, _ = self.envs.step(a)
                self.agent.buffer.push(s, a, r, s_, terminated, probs)
                s = s_

            # train
            value_loss, policy_loss, entropy_loss, approx_kl = self.agent.learn()

            # test
            step += self.sample_num * self.num_envs
            if update % self.test_interval == 0:
                G = self.test()
                if self.log:
                    self.writer.add_scalar('charts/return', G, step)
                    self.writer.add_scalar('charts/lr', self.agent.optimizer.param_groups[0]['lr'], step)
                    self.writer.add_scalar('loss/approx_kl', approx_kl, step)
                    self.writer.add_scalar('loss/value_loss', value_loss, step)
                    self.writer.add_scalar('loss/policy_loss', policy_loss, step)
                    self.writer.add_scalar('loss/entropy_loss', entropy_loss, step)
                self.logger.log(
                    f'update: {update:4d} return: {G: 10.4f} '
                    f'loss: {value_loss:10.4f} {policy_loss:8.4f} {entropy_loss:8.4f} '
                    f'kl: {approx_kl:.8f}')
        # end

    def test(self):
        G = 0

        # test an episode
        s, _ = self.env.reset()
        while True:
            a = self.agent.take_action_test(s)
            s, r, terminated, truncated, _ = self.env.step(a)
            G += r
            if terminated or truncated:
                break

        return G
