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

# Implementation inspired by: https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
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


batch_size = 256
gamma = 0.999 # --> discounted rate
eps_start = 1 # --> Epsilon start
eps_end = 0.01 # --> Epsilon end
eps_decay = 0.001 # --> rate of Epsilon decay
target_update = 10 # --> For every 10 episode, we're going to update 
memory_size = 100000
lr = 0.001
num_episodes = 1000

env = gym.make("CartPole-v1", render_mode = 'human')
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, action_space, device)
memory = ReplayMemeory(memory_size)

policy_net = DQN(observation_space, action_space).to(device)
target_net = DQN(observation_space, action_space).to(device)

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
episode_duration = []

for episode in range(num_episodes):
    state,_ = env.reset()
    for timestep in count():
        env.render()
        action = agent.select_action(torch.from_numpy(state), policy_net)
        (next_state, reward, terminated, done,_) = env.step(action.item())
        memory.push(Experience(torch.from_numpy(state), action, torch.from_numpy(next_state), torch.Tensor([reward])))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, next_states, rewards = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = rewards + (gamma * next_q_values)
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if done:
            episode_duration.append(timestep)
#            plot(episode_duration, 100, env) # plot the duration by 100 moving average
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if get_moving_average(100, episode_duration)[-1] >= 195:
        break
env.close()

