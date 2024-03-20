import argparse
import Q_learning

def main():
    parser = argparse.ArgumentParser(description="run Q_learning")
    parser.add_argument("--human-render", action="store_true")
    parser.add_argument("--experience-replay", action="store_true")
    args = parser.parse_args()

    print("possible commands:")
    print("--human-render: enable visualization")
    print("--experience replay: disable replay buffer")

    # Wait for user input to continue
    input("Press Enter to continue...")

    # print("Model Specifics:") # maybe add summary of model

    if args.human_render:
        env = Q_learning.gym.make("CartPole-v1", render_mode='human')
    else:
        env = Q_learning.gym.make("CartPole-v1")

    if args.experience_replay:
        memory_size = 0  # disable?
    else:
        memory_size = 100000  

    Q_learning.main(env, memory_size) 

if __name__ == "__main__":
    main()
