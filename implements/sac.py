from replay_buffer import ReplayBuffer

from typing import Tuple

import torch
from torch import Tensor, tensor, float32, uint8
from torch import nn
from torch.optim import Adam
from torch.nn.functional import tanh
from torch.distributions import Normal

class SAC:
    def __init__(
            self, 
            env, 
            policy: nn.Module, 
            qf1: nn.Module, 
            qf2: nn.Module, 
            vf: nn.Module, 
            smooth_vf: nn.Module,
            pool_size: int,
            tau: float,
            lr: float,
            scale_reward: float,
            discount: float,
            episode_num: int,
            batch_size: int
    ):
        self._device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        print(f'training start with device {self._device}.')

        self._epsilon = 1e-6

        self._env = env
        self._policy = policy.to(device=self._device)
        self._qf1 = qf1.to(device=self._device)
        self._qf2 = qf2.to(device=self._device)
        self._vf = vf.to(device=self._device)
        self._smooth_vf = smooth_vf.to(device=self._device)
        self._pool = ReplayBuffer(env, pool_size)
        self._tau = tau
        self._lr = lr
        self._scale_reward = scale_reward
        self._discount = discount
        self._episode_num = episode_num
        self._batch_size = batch_size

        self.episode_rewards = []

        # Initialize smooth value function
        for target_param, param in zip(self._smooth_vf.parameters(), self._vf.parameters()):
            target_param.data.copy_(param.data)

    def _sample_actions(
            self,
            batch_state: Tensor
    ) -> Tuple[Tensor, Tensor]:
        policy_output = self._policy(batch_state)
        normal = Normal(policy_output, torch.ones(policy_output))
        x_t = normal.rsample()
        actions = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - actions.pow(2)) + self._epsilon)
        

        return actions, log_prob

    # Update qf1, qf2
    def _update_networks(self):
        batch = self._pool.random_batch(self._batch_size)
        
        batch_state = tensor(batch['state'], dtype=float32).to(self._device)
        batch_actions = tensor(batch['actions'], dtype=float32).to(self._device)
        batch_reward = tensor(batch['reward'], dtype=float32).to(self._device)
        batch_done = tensor(batch['done'], dtype=uint8).to(self._device)
        batch_next_state = tensor(batch['next_state'], dtype=float32).to(self._device)

        qf1_optimizer = Adam(self._qf1.parameters(), lr=self._lr)
        qf2_optimizer = Adam(self._qf2.parameters(), lr=self._lr)
        policy_optimizer = Adam(self._policy.parameters(), lr=self._lr)
        vf_optimizer = Adam(self._vf.parameters(), lr=self._lr)

        qf1_t = self._qf1(torch.cat([batch_state, batch_actions], 1))
        qf2_t = self._qf2(torch.cat([batch_state, batch_actions], 1))
        min_qf_t = torch.min(qf1_t, qf2_t)

        # Update qf1, qf2
        vf_next = self._smooth_vf(batch_next_state)
        ys = self._scale_reward * batch_reward + (1 - batch_done) * self._discount * vf_next

        qf1_loss = 0.5 * torch.mean((ys - qf1_t)**2)
        qf2_loss = 0.5 * torch.mean((ys - qf2_t)**2)

        qf1_optimizer.zero_grad()
        qf2_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        qf1_optimizer.step()
        qf2_optimizer.step()

        # Update policy 
        sample_actions, sample_log_pi = self._sample_actions(batch_state)
        qf1_pi = self._qf1(batch_state, sample_actions)
        qf2_pi = self._qf2(batch_state, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = torch.mean(sample_log_pi - min_qf_pi)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Update vf
        vf_t = self._vf(batch_state)
        log_pi = torch.log(self._policy(batch_state))

        vf_loss = 0.5 * torch.mean((vf_t - (min_qf_t - log_pi))**2)

        vf_optimizer.zero_grad()
        vf_loss.backward()
        vf_optimizer.step()

    def _smooth_target(self):
        for target_param, param in zip(self._smooth_vf.parameters(), self._vf.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)

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
                state_tensor = tensor(state, dtype=float32).to(self._device).unsqueeze(0)
                actions_pred = self._policy(state_tensor) # 3
                actions = tanh(actions_pred).to('cpu').detach().numpy()[0]
                next_state, reward, terminated, truncated, info = self._env.step(actions) # 5

                self._pool.add_sample(state, actions, reward, terminated or truncated, next_state)
                episode_reward += reward
            self._env.close()

            self.episode_rewards.append(episode_reward)
            self._evaluate()

            # Gradient step
            self._update_networks()
            self._smooth_target()
        print('train complete')

        return self.episode_rewards