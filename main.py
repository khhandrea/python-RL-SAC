import gymnasium as gym
import mujoco_py
from matplotlib import pyplot as plt

from typing import Tuple

class SAC:
    def __init__(self, env, policy, qf1, qf2, vf, tau):
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.vf = vf
        self.tau = tau

    def train(self) -> float:
        observation, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
        env.close()
        print('train complete')

    def evaluate(self):
        print('result!')

    def achieve_goal(self) -> bool:
        return False

if __name__ == '__main__':
    MAX_EPISODES = 10

    env = gym.make('Ant-v4')
    policy = None
    qf1 = None
    qf2 = None
    vf = None
    tau = None

    sac = SAC(
        env, 
        policy,
        qf1,
        qf2,
        vf,
        tau
    )

    train_rewards = []
    for episode in range(1, MAX_EPISODES+1):
        # Train
        sac.train()

        # Evaluation
        sac.evaluate()
        if sac.achieve_goal():
            break

    # Show
    env = gym.make('Ant-v4', render_mode='human')

    observation, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    env.close()

    # Plot
    plt.plot(train_rewards)
    plt.show()