import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import numpy as np
import random

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
        self.current_step = 0

    def get_exploration_rate(self) -> float:
        self.current_step += 1
        return self.end + (self.start - self.end) *\
                np.exp(-1. * self.current_step * self.decay)

class FixedEpsilon:
    def __init__(self,epsilon):
        self.epsilon = epsilon
    
    def get_exploration_rate(self):
        return self.epsilon

class Agent:
    def __init__(self, strategy, num_actions, device) -> None:
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net) -> float:
        epsilon = self.strategy.get_exploration_rate()
        state = torch.from_numpy(state)
        # select an action
        if epsilon > np.random.random(): 
            action = np.random.randint(0,self.num_actions)
            return torch.tensor(action).to(device=self.device) # explore
        else:
            with torch.no_grad(): 
                return torch.argmax(policy_net(state)).to(device=self.device) # exploit
                       
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
