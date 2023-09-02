from torch import Tensor, tensor
from torch import nn
from torch.nn.functional import softmax
from torch.distribution import Categorical

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
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.pool_size = pool_size
        self.tau = tau
        self.lr = lr
        self.scale_reward = scale_reward
        self.discount = discount
        self.episode_num = episode_num
        self.batch_size = batch_size

    def _update_critic(self):
        print('update critic(q1, q2)')

    def _update_actor(self):
        print('update actor(V, pi)')

    def _smooth_target(self):
        print('smooth target(V)')

    def _evaluate(self):
        print('result!')

    def _achieve_goal(self) -> bool:
        return False

    def train(self) -> float:
        terminated = truncated = False
        episode_rewards = []

        # At each episode
        for episode in range(self.episode_num):
            state, info = self.env.reset()
            episode_reward = 0

            # At each step
            while not (terminated or truncated):
                # 1. value = critic(state)
                # 2. action = critic(state)
                # 3. policy = actor(state)
                # 4. action = policy.sample()
                # 5. next_state, reward = env.step(action)
                # 6. value_next = critic(next_state)
                # 7. minimize loss

                state_tensor = tensor(state).unsqueeze(0)

                _qf1_t = self.qf1(state_tensor)
                _qf2_t = self.qf2(state_tensor)
                _vf_t = self.vf(state_tensor)
                action_pred = self.policy(state_tensor) # 3
                action_prob = softmax(action_pred, dim=-1)
                dist = Categorical(action_prob)
                action = dist.sample() # 4
                # action = self.env.action_space.sample() # 4
                state, reward, terminated, truncated, info = self.env.step(action) # 5
                # 6
                # 7

                episode_reward += reward
            self.env.close()

            episode_rewards.append(episode_reward)
            self._evaluate()

            # Gradient step
            for gradient_step in range(self.batch_size):
                self._update_critic()
                self._update_actor()
                self._smooth_target()
        print('train complete')

        return episode_rewards