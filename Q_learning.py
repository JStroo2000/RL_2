import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Implementation inspired by: https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

# DQN agent with:
# – Different exploration strategies (at least two)
# – Experience replay (train with a replay buffer)
# – Target network (use another network to provide the update target)
# • Tune hyper-parameters:
# – network architecture (number of layers, number of neurons)
# – learning rate
# – exploration factor
# – The three above are just examples. Think about which are relevant, reason in your report why
# you choose them and what their effect on learning is.
# • Ablation Study: compare different models in terms of learning speed, performance, stability(− here
# means part of the model is removed, either experience replay(ER) or target network(TN) or both.):
# – Compare DQN with DQN−ER
# – Compare DQN with DQN−TN
# – Compare DQN with DQN−EP−TN

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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size

class exploration:
    def __init__(self, policy, start=1, end=0.01, decay=0.001, temp=0.1, q_values=[0,0]):
        self.start = start
        self.end = end
        self.decay = decay
        self.temp = temp
        self.policy = policy
        self.q_values = q_values

    def get_exploration_rate(self, current_step: int) -> float:
        if self.policy == 'egreedy':
            return self.end + (self.start - self.end) *\
                math.exp(-1. * current_step * self.decay)
            # return 0.9
        elif self.policy == 'boltzmann':
            x = self.q_values/self.temp
            z = x - max(x)
            softmax = np.exp(z)/sum(np.exp(z))
            return min(softmax)



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
            action = torch.tensor([action]).to(device=self.device)
            return action # explore

        else:
            with torch.no_grad(): 
                action = torch.argmax(policy_net(state.to(self.device)),dim=-1)
                return action


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
    @staticmethod
    def get_current(policy_net, states, actions, device):
        return policy_net(states.to(device)).gather(1, actions.unsqueeze(-1))
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states, device):
        final_states_location = next_states.flatten(start_dim=1)\
          .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_states_locations = (final_states_location == False)
        non_final_states = next_states[non_final_states_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(device)
        with torch.no_grad():
            values[non_final_states_locations] = target_net(non_final_states).max(dim=1).values
        return values


def eval_policynet(env,policy_net, episode, device):
        eval_rewards = []
        for i in range(5):
            next_state,_ = env.reset()
            episode_reward=0 
            while True:  
                state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                action = torch.argmax(policy_net(state),dim=-1)
                (next_state, eval_reward, eval_terminated, eval_truncated,_) = env.step(action.item())
                episode_reward += eval_reward
                env.render()
                if eval_terminated or eval_truncated:
                    eval_rewards.append(episode_reward)
                    break
        print(f'Reward after {episode} episodes: {np.mean(eval_rewards)}')

def main(env, include_replaybuffer, include_targetnetwork): #-> add include_replay and include_Targetnetwork
    
    #assert (exploration_policy == 'egreedy' or exploration_policy == 'boltzmann'), "exploration policy should be egreedy or boltzmann"
    exploration_policy='egreedy'
    batch_size = 32
    gamma = 0.999 # --> discounted rate
    eps_start = 0.9 # --> Epsilon start
    eps_end = 0.05 # --> Epsilon end
    eps_decay = 0.0001 # --> rate of Epsilon decay
    temp = 0.1 # --> boltzmann policy temperature
    target_update = 100 # --> For every 10 episode, we're going to update 
    memory_size = 200
    lr = 0.0001
    lr_target = 0.1
    num_episodes = 2500
    eval_rate = 100
    current_q_values = np.array([0,0])
    eval_env = gym.make("CartPole-v1")
    env = env
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strategy = exploration(exploration_policy, eps_start, eps_end, eps_decay, temp, current_q_values)
    agent = Agent(strategy, action_space, device)
    memory = ReplayMemory(memory_size)

    # in
    # if include_replaybuffer:
    #     memory = ReplayMemory(memory_size)
    # else:
    #     memory= None
        
    # if include_targetnetwork:
    #     target_net = DQN(observation_space, action_space).to(device)    
    # else:
    #     target_net = None    
    target_net = DQN(observation_space, action_space).to(device)    
    
    policy_net = DQN(observation_space, action_space).to(device)
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    # store info 
    episode_rewards = []
    episode_durations = []
    episode_duration_ma = []

    for episode in range(num_episodes):
        if episode%eval_rate==0:
            eval_policynet(eval_env,policy_net,episode, device)
        state,_ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        for timestep in count():
            env.render()
            # print(timestep)           
            action = agent.select_action(state, policy_net)
            (next_state, reward, terminated, truncated,_) = env.step(action.item())
            episode_reward += reward # add up reward for the episode
            reward = torch.tensor([reward], device=device)
            # print(next_state, reward, terminated, done)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)
            state = next_state
#            agent.strategy.q_values = policy_net(state).detach().numpy()
            if include_replaybuffer and memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                batch = Experience(*zip(*experiences))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                states = torch.cat(batch.state)
                actions = torch.cat(batch.action)
                rewards = torch.cat(batch.reward)
                current_q_values = QValues.get_current(policy_net, states, actions,device)
                next_q_values = torch.zeros(batch_size, device=device)
                if include_targetnetwork:
                    next_q_values[non_final_mask] = QValues.get_next(target_net, non_final_next_states,device)
                else:
                    next_q_values[non_final_mask] = QValues.get_next(policy_net, non_final_next_states,device)
                    
                target_q_values = rewards + (gamma * next_q_values) # Bellman eq
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                episode_loss += loss.item() # add the loss for the episode
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()
                optimizer.zero_grad()
                
            if not include_replaybuffer:
                # without replay buffer just manually convert to tensors instead of extract_tensors
                states = torch.from_numpy(state).unsqueeze(0).to(device)
                next_states = torch.from_numpy(next_state).unsqueeze(0).to(device)
                rewards = torch.Tensor([reward]).to(device)
                
                current_q_values = policy_net(states).gather(dim=1, index=action.unsqueeze(-1)) # this is just get_current

                if include_targetnetwork:
                    next_q_values = QValues.get_next(target_net, next_states,device)
                else:
                    next_q_values = QValues.get_next(policy_net, next_states,device)

                target_q_values = reward + (gamma * next_q_values)
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                episode_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                                
            if terminated or truncated: # was 'done', should be 'terminated'
                episode_durations.append(timestep)
                episode_rewards.append(episode_reward)
                # episode_losses.append(episode_loss)
                episode_duration_ma.append(timestep)
                print(f"episode {episode}: reward={episode_reward}, duration={timestep}, exploration rate={strategy.get_exploration_rate(agent.current_step)}")

                #plot(episode_duration_ma, 100, env) # plot the duration by 100 moving average
                break
            
        print(f"episode {episode}: loss={episode_loss}")

        #if include_targetnetwork and episode % target_update == 0:
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*lr_target + target_net_state_dict[key]*(1-lr_target)
        target_net.load_state_dict(target_net_state_dict)
        moving_avg = get_moving_average(100, episode_duration_ma)[-1]
        if moving_avg >= 250:
            if moving_avg < 400:
                strategy = exploration(exploration_policy, 0.01, 0.01, eps_decay, temp, current_q_values)
            else:
                break
            # break

    env.close()

    # make function, plot after each run.
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
