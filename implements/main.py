from mlp import MLP
from sac import SAC

import gymnasium as gym
from matplotlib import pyplot as plt

from typing import Tuple

if __name__ == '__main__':
    ENV = 'Ant-v4'
    HEALTHY_Z_RANGE = (0.259, 1.0)

    env = gym.make(ENV, healthy_z_range=HEALTHY_Z_RANGE)

    observation_num = env.observation_space.shape
    action_num = env.action_space.shape
    hidden_layer_num = 256

    policy = MLP(27, hidden_layer_num, hidden_layer_num, 8)
    qf1 = MLP(27, hidden_layer_num, hidden_layer_num, 1)
    qf2 = MLP(27, hidden_layer_num, hidden_layer_num, 1)
    vf = MLP(27, hidden_layer_num, hidden_layer_num, 1)
    pool_size = 1e6
    tau = 0.005
    lr = 3e-4
    scale_reward = 20
    discount = 0.99
    episode_num = 1_000_000
    batch_size = 256

    sac = SAC(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        pool_size=pool_size,
        tau=tau,
        lr=lr,
        scale_reward=scale_reward,
        discount=discount,
        episode_num=episode_num,
        batch_size=batch_size
    )

    train_rewards = sac.train()

    # Demonstrate
    env = gym.make(ENV, healthy_z_range=HEALTHY_Z_RANGE, render_mode='human')

    observation, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    env.close()
    print(info)

    # Plot
    plt.plot(train_rewards)
    plt.show()