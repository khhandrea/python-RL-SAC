import gymnasium as gym
import numpy as np

class ReplayBuffer:
    def __init__(
            self, 
            state_num: int,
            action_num: int,
            max_buffer_size: int):

        self._max_buffer_size = max_buffer_size

        self._state = np.zeros((max_buffer_size, state_num))
        self._actions = np.zeros((max_buffer_size, action_num))
        self._rewards = np.zeros(max_buffer_size)
        self._dones = np.zeros(max_buffer_size, dtype='uint8')
        self._next_state = np.zeros((max_buffer_size, state_num))

        self._top = 0
        self._size = 0

    def add_sample(
            self,
            state: np.ndarray,
            actions: np.ndarray,
            reward: float,
            done: bool,
            next_state: np.ndarray
    ):
        self._state[self._top] = state
        self._actions[self._top] = actions
        self._rewards[self._top] = reward
        self._dones[self._top] = done
        self._next_state[self._top] = next_state

        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(
            self,
            batch_size: int
    ):
        indices = np.random.randint(0, self._size, batch_size)
        return {
            'state': self._state[indices],
            'actions': self._actions[indices],
            'reward': self._rewards[indices],
            'done': self._dones[indices],
            'next_state': self._next_state[indices]
        }

    @property
    def size(self):
        return self._size