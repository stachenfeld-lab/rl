

import numpy as np
import gymnasium as gym

class PessimisticQLearningAgent:
    def __init__(self,
                 env: gym.Env,
                 epsilon: float,
                 step_size: float = 0.1,
                 discount_factor: float = 0.99,
                 w: float = 1.0):
        
        # Get number of states and actions from the environment spec.
        self._num_states = env.observation_space.n
        self._num_actions = env.action_space.n
    
        # Create the table of Q-values, all initialized at zero.
        self._q = np.zeros((self._num_states, self._num_actions))
        
        # Containers you may find useful.
        self._epsilon = epsilon
        self._step_size = step_size
        self._discount_factor = discount_factor
        self._w = w

    @property
    def q_values(self):
        return self._q

    def epsilon_greedy(self, q_values_at_s: np.ndarray, epsilon: float):
        """Return an epsilon-greedy action sample."""
        if epsilon < np.random.random():
            # Greedy: Pick action with the largest Q-value.
            return np.argmax(q_values_at_s)
        else:
            # Get the number of actions from the size of the given vector of Q-values.
            num_actions = np.array(q_values_at_s).shape[-1]
            return np.random.randint(num_actions)

    def select_action(self, observation):
        """select the action based on the epsilon greedy rule """
        return self.epsilon_greedy(self._q[observation], self._epsilon)

    def pessimism(self, arr, w):
        """Pessimistic choice rule."""
        return np.min(arr) + w * (np.max(arr) - np.min(arr))

    def update(self, state, action, reward, next_state):
        """Update the Q-value using the pessimistic Q-learning update rule."""
        pessimistic_value = self.pessimism(self._q[next_state], self._w)
        td_target = reward + self._discount_factor * pessimistic_value
        td_error = td_target - self._q[state][action]
        self._q[state][action] += self._step_size * td_error

    def initialize_episode(self, env):
        """Initialize a new episode."""
        state, _ = env.reset()
        return state
