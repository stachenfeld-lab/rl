# Demo test environments 

## Work in progress!

### bandit_demo.py
Run as script from python console. 
Demonstrates how to create a custom bandit environment (see `class DemoBandit(BanditEnv)`) and register it with gymnasium.

The instantiated the environment (`env = gym.make("DemoBandit-v0"...`) can be used by your model de jour. 

When instantiating, you can overwrite the default Bandits by using `p_dist` and `r_dist` to set the bandits' reward probabilities and magnititudes, respectively, as well as the number of bandits (inferred from the length of `p_dist` and `r_dist`). 

The remaining part just prints out the first 100 steps of the environment given either random (`make_rand_action`) or fixed (`non_rand_action`) choice.

### griddemo.py
Defines classes that create a simple Grid World with a target, wall, door, and optional key. 
When called from the command line, a human interface opens and agent can be controlled with arrow keys (turning right/left and moving forward), space (open door), and tab (pick up/drop) key.

Daniel Kimmel, 12 July 2024
