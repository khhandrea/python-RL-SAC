from replay_buffer import ReplayBuffer

from typing import Tuple

from gymnasium import Env
import numpy as np
import torch
from torch import tensor, float32, uint8
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

class SAC:
    def __init__(
            self, *,
            env: Env, 
            policy: nn.Module, 
            qf1: nn.Module, 
            qf2: nn.Module, 
            smooth_qf1: nn.Module, 
            smooth_qf2: nn.Module,
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
        self._smooth_qf1 = smooth_qf1.to(device=self._device)
        self._smooth_qf2 = smooth_qf2.to(device=self._device)
        self._pool = ReplayBuffer(env, pool_size)
        self._tau = tau
        self._lr = lr
        self._alpha = 1. / scale_reward
        self._discount = discount
        self._batch_size = batch_size
        self._start_step = start_step
        self._num_step = num_step
        self._evaluate_episode = evaluate_episode
        self._evaluate_term = evaluate_term

        self.episode_rewards = []

        # Initialize smooth Q function
        for target_param, param in zip(self._smooth_qf1.parameters(), self._qf1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self._smooth_qf2.parameters(), self._qf2.parameters()):
            target_param.data.copy_(param.data)

        self._qf1_optimizer = Adam(self._qf1.parameters(), lr=self._lr)
        self._qf2_optimizer = Adam(self._qf2.parameters(), lr=self._lr)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=self._lr)

        self._writer = SummaryWriter()

        self._total_step = 0
        self._total_episode = 0

    def train(self) -> None:
        # At each episode
        try:
            while self._total_step < self._num_step:
                state, info = self._env.reset()
                terminated = truncated = False
                episode_reward = 0

                # At each step
                while not (terminated or truncated):

                    # Environment step (default: 1)
                    if self._total_step < self._start_step:
                        actions = self._env.action_space.sample()
                    else:
                        state_tensor = tensor(state, dtype=float32).to(self._device).unsqueeze(0)
                        actions, _ = self._policy.select_actions(state_tensor)
                        actions = actions.detach().cpu().numpy()[0]

                    next_state, reward, terminated, truncated, info = self._env.step(actions)

                    self._pool.add_sample(state, actions, reward, terminated, next_state)
                    state = next_state
                    episode_reward += reward
                    self._total_step += 1

                    # Gradient step (default: 1)
                    if self._pool.size >= self._batch_size:
                        self._update_networks()
                        self._smooth_target()

                self._total_episode += 1
                self._writer.add_scalar('reward', episode_reward, self._total_episode)
                if (self._total_episode % self._evaluate_term == 0) and self._total_step > self._start_step :
                    print(f'Episode {self._total_episode:>4d} end. ({self._total_step:>5d} steps)')
                    self._evaluate()
        except Exception as e:
            print('train incomplete with error:')
            print(e)
        else:
            print('train complete')
        finally:
            self._writer.close()
            torch.save({'policy_state_dict': self._policy.state_dict(),
                    'q1_state_dict': self._qf1.state_dict(),
                    'q2_state_dict': self._qf2.state_dict(),
                    'smooth_q1_state_dict': self._smooth_qf1.state_dict(),
                    'smooth_q2_state_dict': self._smooth_qf2.state_dict(),
                    'q1_optimizer_state_dict': self._qf1_optimizer.state_dict(),
                    'q2_optimizer_state_dict': self._qf2_optimizer.state_dict(),
                    'policy_optimizer_state_dict': self._policy_optimizer.state_dict()}, './models.pth')
            print('save completely')

    def _update_networks(self) -> None:
        batch = self._pool.random_batch(self._batch_size)
        
        batch_state = tensor(batch['state'], dtype=float32).to(self._device)
        batch_actions = tensor(batch['actions'], dtype=float32).to(self._device)
        batch_reward = tensor(batch['reward'], dtype=float32).to(self._device)
        batch_done = tensor(batch['done'], dtype=uint8).to(self._device)
        batch_next_state = tensor(batch['next_state'], dtype=float32).to(self._device)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self._policy.select_actions(batch_next_state)
            qf1_next_target = self._smooth_qf1(torch.cat([batch_next_state, next_state_action], 1))
            qf2_next_target = self._smooth_qf2(torch.cat([batch_next_state, next_state_action], 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * next_state_log_pi
            next_q_value = batch_reward +  self._discount * (1 - batch_done) * min_qf_next_target[:, 0]
        qf1_t = self._qf1(torch.cat([batch_state, batch_actions], 1))[:, 0]
        qf2_t = self._qf2(torch.cat([batch_state, batch_actions], 1))[:, 0]

        # Update qf1, qf2
        qf1_loss = 0.5 * F.mse_loss(qf1_t, next_q_value)
        qf2_loss = 0.5 * F.mse_loss(qf2_t, next_q_value)

        self._qf1_optimizer.zero_grad()
        self._qf2_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self._qf1_optimizer.step()
        self._qf2_optimizer.step()

        # Update policy
        sample_actions, sample_log_pi = self._policy.select_actions(batch_state)
        qf1_pi = self._qf1(torch.cat([batch_state, sample_actions], 1))
        qf2_pi = self._qf2(torch.cat([batch_state, sample_actions], 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = torch.mean(self._alpha * sample_log_pi - min_qf_pi)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # return qf1_loss, qf2_loss, policy_loss
        self._writer.add_scalar('qf1_loss', qf1_loss, self._total_step)
        self._writer.add_scalar('qf2_loss', qf2_loss, self._total_step)
        self._writer.add_scalar('policy_loss', policy_loss, self._total_step)

    def _smooth_target(self):
        for target_param, param in zip(self._smooth_qf1.parameters(), self._qf1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) + param.data * self._tau)
        for target_param, param in zip(self._smooth_qf2.parameters(), self._qf2.parameters()):
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