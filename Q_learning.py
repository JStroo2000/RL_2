import gymnasium as gym
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from itertools import count


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


dqn = DQN(n_observations, n_actions).to(device)

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemeory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience) -> None:
        """
        | This function responsible for adding experience to the
        | memory. Also used for sampling experiences from replay memory.
          IF memory less than memory initialied capacity,
            we're going to append inside the memory
          IF NOT
            we're going to begin push new experience onto the front
            of memory overwriting the oldest experience.

          Args:
            experience
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
            self.push_count += 1

    def sample(self, batch_size: int):
        """Sample is equal to the `batch_size` sent to this function`"""
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        """
        |It tells us whether or not we can sample from memory.
          We call that the size of a sample we'll obtain from memory
          will be equal to the batch size we use to train our network.
          For example, suppose we only have 20 experiences in replay memory
          and that our batch size is 50. Then, we will be unable to sample
          because we do not have 20 experiences yet.
        """
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
        """
          strategy: represent an instance from EpsilonGreedyStrategy
          num_actions: corresponding to how many possible actions
                       can the agent take from a given state.
        """
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0
        self.device = device

    def select_action(self, state, policy_net) -> float:
        """
        (UPDATE!)
          The shape of our states have changed, we need to reshape the results
           returned by the network by adding one additional dimension to
           the tensor using the `unsqueeze()` function. 
        """

        epsilon_rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        # select an action
        if epsilon_rate > random.random(): # used random instead of uniform since
                                               # I don't need to end up selecting 1
                                               # at the begining.
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(device=self.device) # explore
        else:
            with torch.no_grad(): # we don't need to compute the graph (aka. gradiant track)
                                # we're just using the model for inference testing.
                # return policy_net(state).argmax(dim=1).to(self.device) # exploit
                return policy_net(state).\
                       unsqueeze(dim=0).\
                       argmax(dim=1).\
                       to(device=self.device) # exploit


def extract_tensors(experiences: NamedTuple) -> Tuple[torch.TensorType]:
    """
    accepts a batch of Experiences and first transposes
    it into an Experience of batches.
    """
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))
    # t_states = torch.cat(batch.state)
    t_states = torch.stack(batch.state)
    t_actions = torch.cat(batch.action)
    # t_next_state = torch.cat(batch.next_state)
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
        """
        | his function accepts a `policy_net`, `states`, and `actions`.
        | When we call this function in our main program, recall that these `states`
          and `actions` are the state-action pairs that were sampled from replay memory.
          So, the states and actions correspond with each other. 
        """
        # Ensure that each prediction corresponting to each action in the batch
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        """
        | This function accepts a `target_net` and `next_states`.
        | Recall that for each next state, we want to obtain the maximum 
          q-value predicted by the `target_net` among all possible next actions. 
        | Link to the description: https://deeplizard.com/learn/video/ewRw996uevM#:~:text=this%20function%20accepts%20a%20target_net%20and%20next_states.%20recall%20that%20for%20each%20next%20state%2C%20we%20want%20to%20obtain%20the%20maximum%20q-value%20predicted%20by%20the%20target_net%20among%20all%20possible%20next%20actions.%20
        """
        # find the locations of all the final states. If an episode is ended by a given action
        ## we're finding the locations of these final states so that 
        ## we know not to pass them to the target_net for q-value predictions
        ## when we pass our non-final next states.
        final_states_location = next_states.flatten(start_dim=1)\
          .max(dim=1)[0].eq(0).type(torch.bool) #check each individual next state tensor to find its maximum value.
          ## If its maximum value is equal to `0`, then we know that this particular next state
          ## is a final state, and we represent that as a `True` within this `final_state_locations` tensor.
          ## next_states that are not final are represented by a `False` value in the tensor.
        non_final_states_locations = (final_states_location == False)

        # Now that we know the locations of the non-final states,
        ## we can now get the values of these states by indexing into the `next_states` tensor
        ## and getting all of the corresponding non_final_states.
        non_final_states = next_states[non_final_states_locations]

        # Next, we find out the batch_size by checking to see how many next states
        ## are in the `next_states` tensor. Using this, we create a new tensor of `zeros`
        ## that has a length equal to the batch size.
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)

        # We then index into this tensor of zeros with the `non_final_state_locations`,
        ## and we set the corresponding values for all of these locations
        ## equal to the maximum predicted `q-values` from the `target_net` across each action.
        values[non_final_states_locations] = target_net(non_final_states).max(dim=1)[0].detach()

        ## torch.Tensor.detach() –  creates a tensor that shares storage with tensor that does not require grad.

        # Return maximum predicted q-value across all actions for each non-final state.
        ## (aka. Maximum Expected rate)
        return values


batch_size = 256
gamma = 0.999 # --> discounted rate
eps_start = 1 # --> Epsilon start
eps_end = 0.01 # --> Epsilon end
eps_decay = 0.001 # --> rate of Epsilon decay
target_update = 10 # --> For every 10 episode, we're going to update 
                         ## the target network with the policy network weights
# 1. Initialize replay memory capacity. 
memory_size = 100000
lr = 0.001
num_episodes = 1000

env = gym.make("CartPole-v1")
action_space = env.action_space
observation_space = env.observation_space.shape[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, action_space, device)
memory = ReplayMemeory(memory_size)


# 2. Initialize the policy network with random weights. 
# policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
policy_net = DQN(observation_space).to(device)

# 3. Clone the policy network, and call it the target network. 
# target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(observation_space).to(device)

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

# Training

episode_duration = []

# 4. For each episode:
for episode in range(num_episodes):
    state = env.reset()

    # (4.2) For each time step:
    for timestep in count():
        """
        (UPDATE!)
          We've now added a line to our main program's nested timestep loop to `render()`
           the environment to the screen. Previously, the call to `render()` was nested
           inside one of the `CartPoleEnvManager`'s screen processing functions 
           `get_processed_screen()` that we no longer utilize. 
        """
        env.render()
        # (4.2.1) Select an action.
        action = agent.select_action(state, policy_net)
        # (4.2.2) Execute selected action in an emulator.
        # (4.2.3) Observe reward and next state.
        (next_state, reward, terminated, done,_) = env.step(action)
        # (4.2.4) Store experience in replay memory.
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        # (4.2.5) Sample random batch from replay memory.
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            # (4.2.6) Preprocess states from batch.
            states, actions, next_states, rewards = extract_tensors(experiences)
            # (4.2.7) Pass batch of preprocessed states to policy network.
            # Get the current Q-values of the policy net and the next Q-values for the target nework 
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            # calculate the target Q-values (aka. Bellman Optimality Equation)
            target_q_values = rewards + (gamma * next_q_values)

            # (4.2.8) Calculate loss between output Q-values and target Q-values.
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            # (4.2.9) Gradient descent updates weights in the policy network to minimize loss.
            ## Calculate the gradients of the loss
            loss.backward()
            ## Update the network with the gradients
            optimizer.step()
            ## Reset the gradients weights & biases before back propagation
            optimizer.zero_grad()

        # Check if the agent took the last action in the episode 
        if done:
            episode_duration.append(timestep)
            plot(episode_duration, 100, env) # plot the duration by 100 moving average
            break

    # (4.2.9.1) After *x* time steps,
    ## weights in the target network are updated to the weights in the policy network.
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    """
    (UPDATE!)
    We've added one line to the end of the episode loop to end the program
     if the network solves the environment by reaching the `100-episode` average reward
     of 195 or higher. 
    """
    if get_moving_average(100, episode_duration)[-1] >= 195:
        break

env.close()



"""
n=500

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation,reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()


"""