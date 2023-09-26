from agent import Agent
from mlp import MLP, PolicyMLP
from replay_buffer import ReplayBuffer

from numpy import ndarray
import torch
from torch import tensor, float32, uint8, load  
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from typing import Tuple
from argparse import ArgumentParser

class SAC(Agent):
    def __init__(
            self,
            state_num: int,
            action_num: int,
            args: ArgumentParser,
    ):
        self.__device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        print(f'training start with device {self.__device}.')

        self.__pool_size = args.pool_size
        self.__tau = args.tau
        self.__lr = args.lr
        self.__alpha = 1. / args.scale_reward
        self.__discount = args.discount

        self.__epsilon = 1e-6
        hidden_layer_num = 256

        self.__policy = PolicyMLP(state_num, hidden_layer_num, hidden_layer_num, action_num).to(device=self.__device)
        self.__qf1 = MLP(state_num + action_num, hidden_layer_num, hidden_layer_num, 1).to(device=self.__device)
        self.__qf2 = MLP(state_num + action_num, hidden_layer_num, hidden_layer_num, 1).to(device=self.__device)
        self.__smooth_qf1 = MLP(state_num + action_num, hidden_layer_num, hidden_layer_num, 1).to(device=self.__device)
        self.__smooth_qf2 = MLP(state_num + action_num, hidden_layer_num, hidden_layer_num, 1).to(device=self.__device)

        self.__smooth_target(initialize=True)

        self.__qf1_optimizer = Adam(self.__qf1.parameters(), lr=self.__lr)
        self.__qf2_optimizer = Adam(self.__qf2.parameters(), lr=self.__lr)
        self.__policy_optimizer = Adam(self.__policy.parameters(), lr=self.__lr)


    def update_networks(
            self,
            pool: ReplayBuffer,
            batch_size: int
    ) -> None:
        batch = pool.random_batch(batch_size)
        
        batch_state = tensor(batch['state'], dtype=float32).to(self.__device)
        batch_action = tensor(batch['action'], dtype=float32).to(self.__device)
        batch_reward = tensor(batch['reward'], dtype=float32).to(self.__device)
        batch_done = tensor(batch['done'], dtype=uint8).to(self.__device)
        batch_next_state = tensor(batch['next_state'], dtype=float32).to(self.__device)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.__policy.select_action(batch_next_state)
            qf1_next_target = self.__smooth_qf1(torch.cat([batch_next_state, next_state_action], 1))
            qf2_next_target = self.__smooth_qf2(torch.cat([batch_next_state, next_state_action], 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.__alpha * next_state_log_pi
            next_q_value = batch_reward +  self.__discount * (1 - batch_done) * min_qf_next_target[:, 0]
        qf1_t = self.__qf1(torch.cat([batch_state, batch_action], 1))[:, 0]
        qf2_t = self.__qf2(torch.cat([batch_state, batch_action], 1))[:, 0]

        # Update qf1, qf2
        qf1_loss = 0.5 * F.mse_loss(qf1_t, next_q_value)
        qf2_loss = 0.5 * F.mse_loss(qf2_t, next_q_value)

        self.__qf1_optimizer.zero_grad()
        self.__qf2_optimizer.zero_grad()
        qf1_loss.backward()
        qf2_loss.backward()
        self.__qf1_optimizer.step()
        self.__qf2_optimizer.step()

        # Update policy
        sample_action, sample_log_pi = self.__policy.select_action(batch_state)
        qf1_pi = self.__qf1(torch.cat([batch_state, sample_action], 1))
        qf2_pi = self.__qf2(torch.cat([batch_state, sample_action], 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = torch.mean(self.__alpha * sample_log_pi - min_qf_pi)

        self.__policy_optimizer.zero_grad()
        policy_loss.backward()
        self.__policy_optimizer.step()

        # Smooth target
        self.__smooth_target()

        # return qf1_loss, qf2_loss, policy_loss
        loss_dict = {
            'qf1_loss': qf1_loss,
            'qf2_loss': qf2_loss,
            'policy_loss': policy_loss
        }

        return loss_dict

    def __smooth_target(
            self, 
            initialize: bool = False):
        if initialize:
            for target_param, param in zip(self.__smooth_qf1.parameters(), self.__qf1.parameters()):
                target_param.data.copy_(param.data)
            for target_param, param in zip(self.__smooth_qf2.parameters(), self.__qf2.parameters()):
                target_param.data.copy_(param.data)
        else:
            for target_param, param in zip(self.__smooth_qf1.parameters(), self.__qf1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.__tau) + param.data * self.__tau)
            for target_param, param in zip(self.__smooth_qf2.parameters(), self.__qf2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.__tau) + param.data * self.__tau)

    def select_action(
            self, 
            state: ndarray, 
            evaluate: bool = False):
        state_tensor = torch.tensor(state, dtype=float32).to(self.__device).unsqueeze(0)
        return self.__policy.select_action(state_tensor, evaluate)

    def load(
            self,
            path:str
        ) -> None:
        models = load(path)
        self.__policy.load_state_dict(models['policy_state_dict'])
        self.__qf1.load_state_dict(models['q1_state_dict'])
        self.__qf2.load_state_dict(models['q2_state_dict'])
        self.__smooth_qf1.load_state_dict(models['smooth_q1_state_dict'])
        self.__smooth_qf2.load_state_dict(models['smooth_q2_state_dict'])

    def save(self):
        model_path = './models.pth'
        torch.save({'policy_state_dict': self.__policy.state_dict(),
                    'q1_state_dict': self.__qf1.state_dict(),
                    'q2_state_dict': self.__qf2.state_dict(),
                    'smooth_q1_state_dict': self.__smooth_qf1.state_dict(),
                    'smooth_q2_state_dict': self.__smooth_qf2.state_dict(),
                    'q1_optimizer_state_dict': self.__qf1_optimizer.state_dict(),
                    'q2_optimizer_state_dict': self.__qf2_optimizer.state_dict(),
                    'policy_optimizer_state_dict': self.__policy_optimizer.state_dict()}, model_path)
        print(f'model saved completely on {model_path}')