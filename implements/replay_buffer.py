import gymnasium as gym
import numpy as np

class ReplayBuffer:
    def __init__(
            self, 
            env: gym.Env,
            max_buffer_size: int):
        action_dim = env.action_space.shape
        observation_dim = env.observation_space.shape

        self._max_buffer_size = max_buffer_size

        self._states = np.zeros((max_buffer_size, observation_dim))
        self._actions = np.zeros((max_buffer_size, action_dim))
        self._rewards = np.zeros(max_buffer_size)
        self._next_states = np.zeros((max_buffer_size, observation_dim))

        self._top = 0
        self._size = 0

    def add_sample(
            self,
            state: np.ndarray,
            actions: np.ndarray,
            reward: float,
            next_state: np.ndarray
    ):
        self._state[self._top] = state
        self._actions[self._top] = actions
        self._rewards[self._top] = reward
        self._next_state[self._top] = next_state

        self._top = (self.top + 1) % self._max_buffer_size
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
            'rewards': self._rewards[indices],
            'next_state': self._next_obs[indices]
        }

    @property
    def size(self):
        return self._size