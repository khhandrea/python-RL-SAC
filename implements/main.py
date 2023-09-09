from mlp import MLP, PolicyMLP
from sac import SAC

import gymnasium as gym
from matplotlib import pyplot as plt
from torch import tensor, float32

from typing import Tuple

if __name__ == '__main__':
    ENV = 'Ant-v4'
    HEALTHY_Z_RANGE = (0.2, 1.0)

    env = gym.make(ENV, healthy_z_range=HEALTHY_Z_RANGE)

    observation_num = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]
    hidden_layer_num = 256

    policy = PolicyMLP(observation_num, hidden_layer_num, hidden_layer_num, action_num)
    qf1 = MLP(observation_num + action_num, hidden_layer_num, hidden_layer_num, 1)
    qf2 = MLP(observation_num + action_num, hidden_layer_num, hidden_layer_num, 1)
    vf = MLP(observation_num, hidden_layer_num, hidden_layer_num, 1)
    smooth_vf = MLP(observation_num, hidden_layer_num, hidden_layer_num, 1)

    pool_size = 1_000_000
    tau = 0.005
    lr = 3e-4
    scale_reward = 5
    discount = 0.99
    episode_num = 200
    batch_size = 256

    sac = SAC(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        smooth_vf=smooth_vf,
        pool_size=pool_size,
        tau=tau,
        lr=lr,
        scale_reward=scale_reward,
        discount=discount,
        episode_num=episode_num,
        batch_size=batch_size
    )

    train_rewards, qf1_losses, qf2_losses, policy_losses, vf_losses = sac.train()

    # Demonstrate
    env = gym.make(ENV, healthy_z_range=HEALTHY_Z_RANGE, render_mode='human')
    policy.to('cpu')

    state, info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        state_tensor = tensor(state, dtype=float32).unsqueeze(0)
        action, _ = policy.select_actions(state_tensor, evaluate=True)
        action = action.detach().numpy()[0]
        state, reward, terminated, truncated, info = env.step(action)
    env.close()

    # Plot
    plt.figure(figsize=(8, 10))
    plt.subplot(3, 1, 1)
    plt.title('rewards')
    plt.plot(train_rewards)

    plt.subplot(3, 2, 3)
    plt.title('Q1 loss')
    plt.plot(qf1_losses)

    plt.subplot(3, 2, 4)
    plt.title('Q2 loss')
    plt.plot(qf2_losses)

    plt.subplot(3, 2, 5)
    plt.title('policy loss')
    plt.plot(policy_losses)

    plt.subplot(3, 2, 6)
    plt.title('V loss')
    plt.plot(vf_losses)

    plt.show()