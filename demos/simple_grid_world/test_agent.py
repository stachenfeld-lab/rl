import gym
import simple_discrete_game
import matplotlib.pyplot as plt
from array2gif import write_gif


def random_agent(episodes=10):
    env = gym.make("GoalGrid-v0")
    all_pos = []
    for i in range(5):
        all_frames = []
        env.reset()
        # env.render()
        for e in range(episodes):
            action = env.action_space.sample()
            img, pos, reward, done, _ = env.step(action)
            all_frames.append(img)
            all_pos.append(pos)
            if done:
                break
        write_gif(all_frames, str(i) + ".gif", fps=10)


if __name__ == "__main__":
    random_agent()
