from replay_buffer import ReplayBuffer

from typing import Tuple

from gymnasium import Env
import numpy as np
import torch
from torch import tensor, float32, uint8
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

class SAC:
    def __init__(
            self, *,
            env: Env, 
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
            batch_size: int,
            start_step: int,
            num_step: int,
            evaluate_episode: int,
            evaluate_term: int
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
        self._batch_size = batch_size
        self._start_step = start_step
        self._num_step = num_step
        self._evaluate_episode = evaluate_episode
        self._evaluate_term = evaluate_term

        self.episode_rewards = []

        # Initialize smooth value function
        for target_param, param in zip(self._smooth_vf.parameters(), self._vf.parameters()):
            target_param.data.copy_(param.data)

        self._qf1_optimizer = Adam(self._qf1.parameters(), lr=self._lr)
        self._qf2_optimizer = Adam(self._qf2.parameters(), lr=self._lr)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=self._lr)
        self._vf_optimizer = Adam(self._vf.parameters(), lr=self._lr)

    def train(self) -> float:
        qf1_losses = []
        qf2_losses = []
        policy_losses = []
        vf_losses = []
        total_step = 0
        total_episode = 0

        # At each episode
        try:
            while total_step < self._num_step:
                state, info = self._env.reset()
                terminated = truncated = False
                episode_reward = 0

                # At each step
                while not (terminated or truncated):

                    # Environment step (default: 1)
                    if total_step < self._start_step:
                        actions = self._env.action_space.sample()
                    else:
                        state_tensor = tensor(state, dtype=float32).to(self._device).unsqueeze(0)
                        actions, _ = self._policy.select_actions(state_tensor)
                        actions = actions.detach().cpu().numpy()[0]

                    next_state, reward, terminated, truncated, info = self._env.step(actions)

                    self._pool.add_sample(state, actions, reward, terminated, next_state)
                    state = next_state
                    episode_reward += reward
                    total_step += 1

                    # Gradient step (default: 1)
                    if self._pool.size >= self._batch_size:
                        qf1_loss, qf2_loss, policy_loss, vf_loss = self._update_networks()
                        self._smooth_target()

                        qf1_losses.append(qf1_loss.detach().cpu().item())
                        qf2_losses.append(qf2_loss.detach().cpu().item())
                        policy_losses.append(policy_loss.detach().cpu().item())
                        vf_losses.append(vf_loss.detach().cpu().item())

                total_episode += 1
                self.episode_rewards.append(episode_reward)
                print(f'Episode {total_episode:>4d} ({total_step:>5d} steps)')
                if (total_episode % self._evaluate_term == 0) and total_step > self._start_step :
                    self._evaluate()
        except Exception as e:
            print('train incomplete with error:')
            print(e)
        finally:
            print('train complete')
            return self.episode_rewards, qf1_losses, qf2_losses, policy_losses, vf_losses

    def _update_networks(self):
        batch = self._pool.random_batch(self._batch_size)
        
        batch_state = tensor(batch['state'], dtype=float32).to(self._device)
        batch_actions = tensor(batch['actions'], dtype=float32).to(self._device)
        batch_reward = tensor(batch['reward'], dtype=float32).to(self._device)
        batch_done = tensor(batch['done'], dtype=uint8).to(self._device)
        batch_next_state = tensor(batch['next_state'], dtype=float32).to(self._device)

        qf1_t = self._qf1(torch.cat([batch_state, batch_actions], 1))
        qf2_t = self._qf2(torch.cat([batch_state, batch_actions], 1))
        with torch.no_grad():
            vf_next = self._smooth_vf(batch_next_state)
            ys = self._scale_reward * batch_reward + (1 - batch_done) * self._discount * vf_next
            sample_actions, sample_log_pi = self._policy.select_actions(batch_state)
            min_qf_t = torch.min(qf1_t, qf2_t)

        # Update qf1, qf2
        qf1_loss = 0.5 * F.mse_loss(ys, qf1_t)
        qf2_loss = 0.5 * torch.mean(torch.pow(ys - qf2_t, 2))

        self._qf1_optimizer.zero_grad()
        self._qf2_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self._qf1_optimizer.step()
        self._qf2_optimizer.step()

        # Update policy
        qf1_pi = self._qf1(torch.cat([batch_state, sample_actions], 1))
        qf2_pi = self._qf2(torch.cat([batch_state, sample_actions], 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = torch.mean(sample_log_pi - min_qf_pi)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()
        
        # Update vf
        vf_t = self._vf(batch_state)
        with torch.no_grad():
            vf_target = min_qf_t - sample_log_pi

        vf_loss = 0.5 * torch.mean(torch.pow(vf_t - vf_target, 2))

        self._vf_optimizer.zero_grad()
        vf_loss.backward()
        self._vf_optimizer.step()

        return qf1_loss, qf2_loss, policy_loss, vf_loss

    def _smooth_target(self):
        for target_param, param in zip(self._smooth_vf.parameters(), self._vf.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)

    def _evaluate(self):
        average_reward = 0.
        for _ in range(self._evaluate_episode):
            episode_reward = 0
            state, info = self._env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                state_tensor = tensor(state, dtype=float32).to(self._device).unsqueeze(0)
                action, _ = self._policy.select_actions(state_tensor, evaluate=True)
                action = action.detach().cpu().numpy()[0]
                state, reward, terminated, truncated, info = self._env.step(action)
                episode_reward += reward
            average_reward += episode_reward
        average_reward /= self._evaluate_episode
        print(f'average reward: {average_reward}')