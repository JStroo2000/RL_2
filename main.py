import argparse
import Q_learning

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
    print("--experience-replay: disable replay buffer")
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
    
    # gridsearch
    search_space = {
    'layer1': [16,32, 64, 128],
    'layer2': [16,32, 64, 128],
    'lr': [0.0001,0.0005, 0.001, 0.005],
    'lr_target': [.2,.1,.05,.01],
    'exploration_strat': ['egreedy_decay','egreedy','boltzman'],
    'gamma': [0.9,0.99, 0.999],
    'batch_size': [16,32,64,128,256],
    'memory_size': [50,100,200,500]
    }


    exp_dict = {'egreedy_decay':[(.9,.0001,.05),(.9,.001,.05),(.9,.0001,.001)],
                    'egreedy':[.9,.95,.99],
                    'boltzman':[.2,.1,.05,.01]}

    Q_learning.main(env, include_replaybuffer, include_targetnetwork, 
                    search_space, exp_dict,50)

if __name__ == "__main__":
    main()
