from gym_bandits.bandit import BanditEnv
import gymnasium as gym
import numpy as np

make_rand_action = True
non_rand_action = 0 # when not taking random action

# Define custom bandit environment
class DemoBandit(BanditEnv):
    """Stochastic version with a large difference between which bandit pays out of two choices"""
    def __init__(self,p_dist=(0.8, 0.2), r_dist=(1, 1)):
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

# register environment
gym.envs.register(
    id='{}-{}'.format('DemoBandit', 'v0'),
    entry_point=DemoBandit,
)

# use package-defined bandit
# env = gym.make("BanditTwoArmedHighLowFixed-v0")

# use custom demo bandit. optionally pass p_dist and r_dist to override default values
env = gym.make("DemoBandit-v0", p_dist = [1, 0.9, 0.1], r_dist=[1,2,10]) # Replace with relevant env

env.reset()
for _ in range(100):

    if make_rand_action:
        _action = np.random.randint(env.action_space.n)
    else:
        _action = non_rand_action

    if make_rand_action:
        output = env.step(_action) # --> obs, rwd, terminated, truncated
    else:
        output = env.step(_action)  # --> obs, rwd, terminated, truncated

    print('Action = {}, Reward = {}'.format(_action, output[1]))

