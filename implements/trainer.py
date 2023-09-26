import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from sac import SAC


from typing import List

class Trainer:
    def __init__(self, args):
        self.__env_name = args.env_name
        self.__healthy_min = args.healthy_min
        self.__start_step = args.start_step
        self.__num_step = args.num_step
        self.__evaluate_episode = args.evaluate_episode
        self.__evaluate_term = args.evaluate_term
        self.__pool_size = args.pool_size
        self.__batch_size = args.batch_size
        self.__discount = args.discount

        HEALTHY_Z_RANGE = (self.__healthy_min, 1.0)
        self.__env = gym.make(self.__env_name, healthy_z_range=HEALTHY_Z_RANGE)
        
        self.__state_num = self.__env.observation_space.shape[0]
        self.__action_num = self.__env.action_space.shape[0]

        self.__sac_agent = SAC(self.__state_num, self.__action_num, args)
        

    def train(self) -> None:
        total_step = 0
        total_episode = 0

        writer = SummaryWriter()
        pool = ReplayBuffer(self.__state_num, self.__action_num, self.__pool_size)

        # At each episode
        while total_step < self.__num_step:
            state, info = self.__env.reset()
            terminated = truncated = False
            rewards = []
            total_episode_step = total_step

            # At each step
            while not (terminated or truncated):

                # Environment step (default: 1)
                if total_step < self.__start_step:
                    action = self.__env.action_space.sample()
                else:
                    action, _ = self.__sac_agent.select_action(state)
                    action = action.detach().cpu().numpy()[0]

                next_state, reward, terminated, truncated, info = self.__env.step(action)

                pool.add_sample(state, action, reward, terminated, next_state)
                state = next_state
                rewards.append(reward)
                total_step += 1

                # Gradient step (default: 1)
                if pool.size >= self.__batch_size:
                    scalar_dict = self.__sac_agent.update_parameters(pool, self.__batch_size)

                    for name in scalar_dict:
                        writer.add_scalar(name, scalar_dict[name], total_step)

            for idx, G in enumerate(self.__get_returns(rewards)):
                writer.add_scalar('return', G, total_episode_step + idx)
            writer.add_scalar('reward', sum(rewards), total_episode)

            total_episode += 1

            if (total_episode % self.__evaluate_term == 0) and total_step > self.__start_step :
                # Evaluate
                average_reward = 0.
                for _ in range(self.__evaluate_episode):
                    episode_reward = 0
                    state, info = self.__env.reset()
                    terminated = truncated = False
                    while not (terminated or truncated):
                        action, _ = self.__sac_agent.select_action(state, evaluate=True)
                        action = action.detach().cpu().numpy()[0]
                        state, reward, terminated, truncated, info = self.__env.step(action)
                        episode_reward += reward
                    average_reward += episode_reward
                average_reward /= self.__evaluate_episode
                print(f'average reward: {average_reward}')
                print(f'Episode {total_episode:>4d} end. ({total_step:>5d} steps)')
                
        print('train complete')
        writer.close()
        
        self.__sac_agent.save()

    def test(self) -> None:
        HEALTHY_Z_RANGE = (self.__healthy_min, 1.0)
        env = gym.make(self.__env_name, healthy_z_range=HEALTHY_Z_RANGE, render_mode='human')

        state, info = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = self.__sac_agent.select_action(state, evaluate=True)
            action = action.detach().cpu().numpy()[0]
            state, reward, terminated, truncated, info = env.step(action)
        env.close()

    def __get_returns(
            self,
            rewards: List[float]
    ) -> List[float]:
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + G * self.__discount
            returns.insert(0, G)

        return returns


    def load_agent(
            self,
            path:str
        ) -> None:
        self.__sac_agent.load(path)