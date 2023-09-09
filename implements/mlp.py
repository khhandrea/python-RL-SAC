from typing import Tuple

import torch
from torch import Tensor
from torch import nn, clamp
from torch.distributions import Normal
from torch.nn.functional import relu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        # Define layers
        self.fc_1 = nn.Linear(input_dim, hidden_dim1)
        self.fc_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_3 = nn.Linear(hidden_dim2, output_dim)

        # Parameter initialize
        for linear in (self.fc_1, self.fc_2, self.fc_3):
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc_1(x)
        x = relu(x)
        x = self.fc_2(x)
        x = relu(x)
        x = self.fc_3(x)
        return x
    
class PolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        # Define layers
        self.fc_1 = nn.Linear(input_dim, hidden_dim1)
        self.fc_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mean = nn.Linear(hidden_dim2, output_dim)
        self.fc_std = nn.Linear(hidden_dim2, output_dim)

        # Parameter initialize
        for linear in (self.fc_1, self.fc_2, self.fc_mean, self.fc_std):
            nn.init.xavier_normal_(linear.weight)
            linear.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.fc_1(x)
        x = relu(x)
        x = self.fc_2(x)
        x = relu(x)
        
        mean = self.fc_mean(x)
        std = clamp(self.fc_std(x), LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, std
    
    def select_actions(
            self, 
            state: Tensor,
            evaluate: bool=False
    ) -> Tuple[Tensor, Tensor]:
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        actions = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log((1 - actions.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        if evaluate:
            actions = torch.tanh(mean)
        # print(actions)
        return actions, log_prob