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
        pass

    def _update_actor(self):
        pass

    def _smooth_target(self):
        pass

    def _evaluate(self):
        pass


    def train(self) -> float:
        episode_rewards = []

        # At each episode
        for episode in range(self.episode_num):
            state, info = self.env.reset()
            terminated = truncated = False
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

                state_tensor = tensor(state, dtype=float32).unsqueeze(0)

                qf1_t = self.qf1(state_tensor)
                qf2_t = self.qf2(state_tensor)
                vf_t = self.vf(state_tensor)
                actions = self.policy(state_tensor) # 3
                actions_normalized = tanh(actions).detach().numpy()[0]
                state, reward, terminated, truncated, info = self.env.step(actions_normalized) # 5
                # 6
                # 7

                episode_reward += reward
            self.env.close()

            episode_rewards.append(episode_reward)
            print('reward: ', episode_reward)
            self._evaluate()

            # Gradient step
            for gradient_step in range(self.batch_size):
                # pool.random_batch
                self._update_critic()
                self._update_actor()
                self._smooth_target()
        print('train complete')

        return episode_rewards