import gymnasium as gym
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agent import Agent, ReplayMemeory,DQN,Experience


# Implementation inspired by: https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

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

def eval_policynet(env,policy_net):
        eval_rewards = []
        for i in range(10):
            state,_ = env.reset()
            while True:  
                episode_reward=0             
                action = torch.argmax(policy_net(torch.from_numpy(state)))
                (next_state, eval_reward, eval_terminated, eval_truncated,_) = env.step(action.item())
                episode_reward +=eval_reward
                eval_env.render()
                if eval_terminated or eval_truncated:
                    eval_rewards.append(episode_reward)
                    break
        print(f'Reward after {episode} episodes: {np.mean(eval_rewards)}')


def main(env, include_replaybuffer, include_targetnetwork,strategy ): #-> add include_replay and include_Targetnetwork
    
    batch_size = 32
    gamma = 0.999 # --> discounted rate
    eps_start = 1 # --> Epsilon start
    eps_end = 0.005 # --> Epsilon end
    eps_decay = 0.001 # --> rate of Epsilon decay
    target_update = 50 # --> For every 10 episode, we're going to update 
    memory_size = 100
    lr = 0.001
    num_episodes = 500
    eval_rate = 100

    eval_env = gym.make("CartPole-v1", render_mode = 'human')
    # env = gym.make("CartPole-v1", render_mode = 'human')
    env = env
    action_space = env.action_space.n
    observation_space = env.observation_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, action_space, device)
    
    memory = ReplayMemeory(memory_size)

    # in
    # if include_replaybuffer:
    #     memory = ReplayMemeory(memory_size)
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
            eval_policynet(eval_env,policy_net)
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
            if terminated:
                next_state,_ = env.reset()
            memory.push(Experience(torch.from_numpy(state), action, torch.from_numpy(next_state), torch.Tensor([reward])))
            state = next_state

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
                                
            if terminated: # was 'done', should be 'terminated'
                episode_durations.append(timestep)
                episode_rewards.append(episode_reward)
                # episode_losses.append(episode_loss)
                episode_duration_ma.append(timestep)
                # print(f"episode {episode}: reward={episode_reward}, duration={timestep}, exploration rate={strategy.get_exploration_rate(agent.current_step)}")

                #plot(episode_duration_ma, 100, env) # plot the duration by 100 moving average
                break
            
        # print(f"episode {episode}: loss={episode_loss}")

        if include_targetnetwork and episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

#        if get_moving_average(100, episode_duration_ma)[-1] >= 195: -> I don't understand what this does, so i took it out.
#            break
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
