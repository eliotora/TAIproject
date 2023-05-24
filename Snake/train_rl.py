import argparse
import sys
from pathlib import Path
from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from agent import Agent, ReinforcementTrainingGame


def training():
    """
    The training of the agent with the snake environment
    """
    parser = argparse.ArgumentParser(
        description="The Snake game trainer for RL."
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to weights file to save to.",
        default="weights.h5",
    )
    parser.add_argument(
        "-e",
        "--episodes",
        type=int,
        help="Number of episodes to train on (default=10000).",
        default=10000,
    )
    parser.add_argument(
        "-rt",
        "--retrain",
        type=str,
        help="Train from a saved file",
    )

    args = parser.parse_args()
    if Path(args.weights).is_file():
        parser.print_help()
        print(
            f"File {args.weights}, already exists, do you want to overwrite ?"
        )
        y = input("Type yes or no: ")
        if y != "yes":
            print("Aborting.")
            sys.exit()

    # --- Initialisation --- #
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    score_distibution = [0 for i in range(200)]
    total_score = 0
    record = 0
    score = 0
    iteration = 0
    if args.retrain:
        agent = Agent(epsilon=0, epsilon_min=0)
        agent.load(args.retrain)
        file_name = args.retrain.split(".")[0]
        values = file_name.split("_")
        print(values)
        record = int(values[1])
        iteration = int(values[2]) + 1
        game = ReinforcementTrainingGame(reward_live=0, reward_eat=10, reward_dead=-10)
    else:
        agent = Agent(gamma=0.9, decay=0.99)
        game = ReinforcementTrainingGame()
    game.start_run()
    for i in tqdm(range(args.episodes)):
        done = False
        move_nbr = 0
        while not done:
            state_old = agent.get_state(game)

            final_move = agent.act_train(state_old)
            agent.direction = final_move

            reward, done, score = game.next_tick(final_move)

            state_new = agent.get_state(game)

            # agent.training_montage(state_old, final_move, reward, state_new, done)

            agent.fill_memory(state_old, final_move, reward, state_new, done)

            move_nbr += 1

        score_distibution[score] += 1
        game.start_run()
        agent.n_games += 1

        # if i % 200 == 0:
        #     agent.model.save(f"try12/{args.weights}_{record}_{iteration}_{i}.h5")

        # agent.train_long_memory()

        # if score > record:
        #     record = score
        #     agent.model.save(f"try12/{args.weights}_{record}_{iteration}.h5")

        # print('Game', agent.n_games, 'Score', score, 'Record', record, 'Move number', move_nbr)
        total_score += score
        mean_score = total_score / agent.n_games

    print(score_distibution)
    print(mean_score)
    agent.model.save(f"try12/{args.weights}_{record}_{iteration}_end.h5")


if __name__ == "__main__":
    training()
