import argparse
import Q_learning
import numpy as np
# – Compare DQN with DQN−ER
# – Compare DQN with DQN−TN
# – Compare DQN with DQN−EP−TN
# -> option to run without replay buffer and/or Target Network
#   Add command here, in Q_learning add variable for true/false to include in main

def main():
    parser = argparse.ArgumentParser(description="run Q_learning")
    parser.add_argument("--human-render", action="store_true")
    parser.add_argument("--experience-replay", action="store_true")
    parser.add_argument("--target-network", action="store_true")

    
    args = parser.parse_args()

    print("possible commands:")
    print("--human-render: enable visualization")
    print("--experience replay: disable replay buffer")
    print("--target-network: disable target network")

    input("Press enter to continue...")

    if args.human_render:
        env = Q_learning.gym.make("CartPole-v1", render_mode='human')
    else:
        env = Q_learning.gym.make("CartPole-v1")

    if args.experience_replay:
        include_replaybuffer = False # disable?
    else:
        include_replaybuffer = True  
        
    if args.target_network:
        include_targetnetwork = False
    else:
        include_targetnetwork = True

    for epsilon in [0.3,0.2,0.1,0.05,0.01,0.005]:
        print(f'epsilon: {epsilon}')
        strategy = Q_learning.FixedEpsilon(epsilon)
        rewards,episode_lengths = Q_learning.main(env, include_replaybuffer, include_targetnetwork,strategy) 
        with open(f'rewards_run_{epsilon}.npy','wb') as f:
            np.save(f, rewards)
        with open(f'episode_length_{epsilon}','wb') as f:
            np.save(f,episode_lengths)
        print(f'mean reward: {np.mean(rewards[-50:])}')


if __name__ == "__main__":
    main()
