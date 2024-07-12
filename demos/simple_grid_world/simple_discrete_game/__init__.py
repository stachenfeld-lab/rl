from gym.envs.registration import register

register(id="GoalGrid-v0", entry_point="simple_discrete_game.envs:GoalGridEnv")
