import gymnasium as gym
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agent import Agent, ReplayMemeory, FixedEpsilon, DQN, Experience
from collections import namedtuple

# Implementation inspired by: https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

def extract_tensors(experiences: namedtuple):

    batch = Experience(*zip(*experiences))
    t_states = torch.stack(batch.state)
    t_actions = torch.cat(batch.action)
    t_next_state = torch.stack(batch.next_state)
    t_rewards = torch.cat(batch.reward)

    return (t_states,
            t_actions,
            t_next_state,
            t_rewards)

class QValues:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_states_location = next_states.flatten(start_dim=1)\
          .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_states_locations = (final_states_location == False)
        non_final_states = next_states[non_final_states_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_states_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def eval_policynet(env,policy_net, episode):
        eval_rewards = []
        for i in range(10):
            state,_ = env.reset()
            episode_reward=0 
            while True:  
                action = torch.argmax(policy_net(torch.from_numpy(state))).unsqueeze(dim=0)
                (next_state, eval_reward, eval_terminated, eval_truncated,_) = env.step(action.item())
                episode_reward += eval_reward
                env.render()
                if eval_terminated or eval_truncated:
                    eval_rewards.append(episode_reward)
                    break
        print(f'Reward after {episode} episodes: {np.mean(eval_rewards)}')


def main(env ,strategy ): #-> add include_replay and include_Targetnetwork
    
    batch_size = 32
    gamma = 0.999 # --> discounted rate
    eps_start = 1 # --> Epsilon start
    eps_end = 0.005 # --> Epsilon end
    eps_decay = 0.001 # --> rate of Epsilon decay
    target_update = 50 # --> For every 10 episode, we're going to update 
    memory_size = 100
    lr = 0.001
    num_episodes = 5000
    eval_rate = 250

    eval_env = gym.make("CartPole-v1", render_mode = 'human')
    env = env
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) -> given in main
    agent = Agent(strategy, action_space, device)
    memory = ReplayMemeory(memory_size)

    target_net = DQN(observation_space, action_space).to(device)    
    policy_net = DQN(observation_space, action_space).to(device)
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    # store info 
    episode_rewards = []
    episode_durations = []
    episode_duration_ma = []

    for episode in range(num_episodes):
        if episode%eval_rate==0:
            eval_policynet(eval_env,policy_net,episode)
        state,_ = env.reset()
        episode_reward = 0
        episode_loss = 0
        for timestep in count():
            env.render()
            action = agent.select_action(state, policy_net)
            (next_state, reward, terminated, truncated,_) = env.step(action.item())
            episode_reward += reward # add up reward for the episode

            memory.push(Experience(torch.from_numpy(state), action.unsqueeze(-1), torch.from_numpy(next_state), torch.Tensor([episode_reward])))

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, next_states, rewards = extract_tensors(experiences)
                current_q_values = policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)) # this is just get_current
                next_q_values = QValues.get_next(policy_net, next_states)
                target_q_values = reward + (gamma * next_q_values)
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                episode_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            state = next_state
            if terminated:
                episode_durations.append(timestep)
                episode_rewards.append(episode_reward)
                # episode_losses.append(episode_loss)
                episode_duration_ma.append(timestep)
                break

        #print(f"episode {episode}: loss={episode_loss}")



    env.close()
    return np.array(episode_rewards),np.array(episode_durations)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    strategy = FixedEpsilon(0.01)
    rewards,episode_lengths = main(env,strategy) 
    print(f'mean reward: {np.mean(rewards[-50:])}')