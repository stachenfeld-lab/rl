import numpy as np
import gymnasium as gym

class QLearningAgent:
    def __init__(self,
                 env: gym.Env,
                 epsilon: float,
                 step_size: float = 0.1,
                 discount_factor: float = 0.99):
        
        # Get number of states and actions from the environment spec.
        self._num_states = env.observation_space.n
        self._num_actions = env.action_space.n
    
        # Create the table of Q-values, all initialized at zero.
        self._q = np.zeros((self._num_states, self._num_actions))
        
        # Containers you may find useful.
        self._epsilon = epsilon
        self._step_size = step_size
        self._discount_factor = discount_factor

    @property
    def q_values(self):
        return self._q

    # Epsilon Greedy Policy 
    def epsilon_greedy(self, q_values_at_s: np.ndarray, epsilon: float = 0.1):
        """Return an epsilon-greedy action sample."""
        if epsilon < np.random.random():
            # Greedy: Pick action with the largest Q-value.
            return np.argmax(q_values_at_s)
        else:
            # Get the number of actions from the size of the given vector of Q-values.
            num_actions = np.array(q_values_at_s).shape[-1]
            return np.random.randint(num_actions)

    def select_action(self, observation):
        return self.epsilon_greedy(self._q[observation], self._epsilon)
    
    def update(self, state, action, reward, next_state):
        """Update the Q-value using the Q-learning update rule."""
        best_next_action = np.argmax(self._q[next_state])
        td_target = reward + self._discount_factor * self._q[next_state][best_next_action]
        td_error = td_target - self._q[state][action]
        self._q[state][action] += self._step_size * td_error

    def initialize_episode(self, env):
        """Initialize a new episode."""
        state, _ = env.reset()
        return state
