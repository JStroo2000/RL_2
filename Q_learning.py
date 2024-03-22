import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Implementation inspired by: https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions,layer1=64,layer2=64,activation=F.relu):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer1)
        self.layer2 = nn.Linear(layer1, layer2)
        self.layer3 = nn.Linear(layer2, n_actions)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        return self.layer3(x)


Experience = namedtuple('Experience',
    ('state', 'action', 'next_state', 'reward')
    )

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)

    if len(values) >= period: 


        moving_avg = values.unfold(dimension=0,
                                   size=period, 
                                   step=1)\
                                   .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()    


class ReplayMemeory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step: int) -> float:
        return self.end + (self.start - self.end) *\
                math.exp(-1. * current_step * self.decay)

class FixedEpsilon:
    def __init__(self,epsilon):
        self.epsilon = epsilon
    
    def get_exploration_rate(self, current_step):
        return self.epsilon

class Agent:
    def __init__(self, strategy, num_actions, device) -> None:
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device

    def select_action(self, state, policy_net) -> float:

        epsilon_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # select an action
        if epsilon_rate > random.random(): 
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device=self.device) # explore
        else:
            with torch.no_grad(): 
                return policy_net(state).\
                       unsqueeze(dim=0).\
                       argmax(dim=1).\
                       to(device=self.device) # exploit


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

def main(env, include_replaybuffer, include_targetnetwork,strategy ): #-> add include_replay and include_Targetnetwork
    
    batch_size = 32
    gamma = 0.999 # --> discounted rate
    eps_start = 1 # --> Epsilon start
    eps_end = 0.005 # --> Epsilon end
    eps_decay = 0.001 # --> rate of Epsilon decay
    target_update = 50 # --> For every 10 episode, we're going to update 
    memory_size = 100
    lr = 0.001
    num_episodes = 5000

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
        state,_ = env.reset()
        episode_reward = 0
        episode_loss = 0
        for timestep in count():
            env.render()
            # print(timestep)
            action = agent.select_action(torch.from_numpy(state), policy_net)
            (next_state, reward, terminated, done,_) = env.step(action.item())
            # print(next_state, reward, terminated, done)
            episode_reward += reward # add up reward for the episode
            memory.push(Experience(torch.from_numpy(state), action, torch.from_numpy(next_state), torch.Tensor([reward])))
            if not include_replaybuffer:
                # without replay buffer just manually convert to tensors instead of extract_tensors
                states = torch.from_numpy(state).unsqueeze(0)
                next_states = torch.from_numpy(next_state).unsqueeze(0)
                rewards = torch.Tensor([reward])
                
                current_q_values = policy_net(states).gather(dim=1, index=action.unsqueeze(-1)) # this is just get_current

                if include_targetnetwork:
                    next_q_values = QValues.get_next(target_net, next_states)
                else:
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
        if include_replaybuffer and memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, next_states, rewards = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            if include_targetnetwork:
                next_q_values = QValues.get_next(target_net, next_states)
            else:
                next_q_values = QValues.get_next(policy_net, next_states)
                
            target_q_values = rewards + (gamma * next_q_values) # Bellman eq
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            episode_loss += loss.item() # aad the loss for the episode
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
        #print(f"episode {episode}: loss={episode_loss}")

        if include_targetnetwork and episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env.close()
    return np.array(episode_rewards),np.array(episode_durations)

    ## plot 
    plt.figure(figsize=(10, 5))

    # plot moving average
    plt.subplot(1, 2, 1)
    plt.plot(get_moving_average(100, episode_rewards))
    plt.xlabel('episode')
    plt.ylabel('moving avg of reward')
    plt.title('moving avg of reward')

    # plot duration
    plt.subplot(1, 2, 2)
    plt.plot(episode_durations)
    plt.xlabel('episode')
    plt.ylabel('timesteps')
    plt.title('duration of episodes')

    plt.tight_layout()
    plt.show()
