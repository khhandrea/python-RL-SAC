import gymnasium as gym
import numpy as np

class ReplayBuffer:
    def __init__(
            self, 
            state_num: int,
            action_num: int,
            max_buffer_size: int):

        self.__max_buffer_size = max_buffer_size

        self.__state = np.zeros((max_buffer_size, state_num))
        self.__action = np.zeros((max_buffer_size, action_num))
        self.__rewards = np.zeros(max_buffer_size)
        self.__dones = np.zeros(max_buffer_size, dtype='uint8')
        self.__next_state = np.zeros((max_buffer_size, state_num))

        self.__top = 0
        self.__size = 0

    def add_sample(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            done: bool,
            next_state: np.ndarray
    ):
        self.__state[self.__top] = state
        self.__action[self.__top] = action
        self.__rewards[self.__top] = reward
        self.__dones[self.__top] = done
        self.__next_state[self.__top] = next_state

        self.__top = (self.__top + 1) % self.__max_buffer_size
        if self.__size < self.__max_buffer_size:
            self.__size += 1

    def random_batch(
            self,
            batch_size: int
    ):
        indices = np.random.randint(0, self.__size, batch_size)
        return {
            'state': self.__state[indices],
            'action': self.__action[indices],
            'reward': self.__rewards[indices],
            'done': self.__dones[indices],
            'next_state': self.__next_state[indices]
        }

    @property
    def size(self):
        return self.__size