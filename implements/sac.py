from replay_buffer import ReplayBuffer

import torch
from torch import Tensor, tensor, float32
from torch import nn
from torch.nn.functional import tanh
from torch.distributions import Categorical

class SAC:
    def __init__(
            self, 
            env, 
            policy: nn.Module, 
            qf1: nn.Module, 
            qf2: nn.Module, 
            vf: nn.Module, 
            pool_size: int,
            tau: float,
            lr: float,
            scale_reward: float,
            discount: float,
            episode_num: int,
            batch_size: int
    ):
        self._env = env
        self._policy = policy
        self._qf1 = qf1
        self._qf2 = qf2
        self._vf = vf
        self._pool = ReplayBuffer(env, pool_size)
        self._tau = tau
        self._lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._episode_num = episode_num
        self._batch_size = batch_size

        self.episode_rewards = []

    # Update qf1, qf2
    def _update_critic(self):
        # batch_state_tensor = tensor(self._batch['state'])
        # vf_next = self._vf(batch_state_tensor)
        # ys = self._scale_reward * self._batch['reward'] + (1 - self._batch['done']) * self._discount * vf_next
        # _td_loss1_t = 0.5 * torch.mean((ys - self.)**2)
        pass

    # Update policy, vf
    def _update_actor(self):
        self._policy
        self._vf
        pass

    def _smooth_target(self):
        pass

    def _evaluate(self):
        print(self.episode_rewards[-1])

    def train(self) -> float:
        # At each episode
        for episode in range(self._episode_num):
            state, info = self._env.reset()
            terminated = truncated = False
            episode_reward = 0

            # At each step
            while not (terminated or truncated):
                state_tensor = tensor(state, dtype=float32).unsqueeze(0)
                actions_pred = self._policy(state_tensor) # 3
                actions = tanh(actions_pred).detach().numpy()[0]
                next_state, reward, terminated, truncated, info = self._env.step(actions) # 5

                self._pool.add_sample(state, actions, reward, terminated or truncated, next_state)
                episode_reward += reward
            self._env.close()

            self.episode_rewards.append(episode_reward)
            self._evaluate()

            # Gradient step
            self._batch = self._pool.random_batch(self._batch_size)

            # end = 1 if (terminated or truncated) else 0
            # next_state_tensor = tensor(next_state, dtype=float32).unsqueeze(0)

            
            # self._min_log_target = torch.min(self._qf1_t, self._qf2_t)
            # self._vf_t = self._vf(state_tensor)
            # self._vf_next_t = self._vf(next_state_tensor)

            self._update_critic()
            self._update_actor()
            self._smooth_target()
        print('train complete')

        return self.episode_rewards