import gymnasium as gym
from tqdm import tqdm

n=500

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation,reward, terminated, truncated, info = env.step(action)
    print("info : ",info);
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()