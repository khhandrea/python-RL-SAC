from torch import Tensor
from torch import nn
from torch.nn.functional import relu, softmax

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
        dropout: float=0.5
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