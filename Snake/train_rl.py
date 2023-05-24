import argparse
import sys
from pathlib import Path
from tqdm import tqdm
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable the spam Warning from TensorFlow

    # score_distibution = [0 for _ in range(200)]
    total_score = 0
    record = 0
    score = 0
    iteration = 0

    if args.retrain:  # In retrain mode we do not allow any random moves
        agent = Agent(epsilon=0, epsilon_min=0)
        agent.load(args.retrain)
        file_name = args.retrain.split(".")[0]
        values = file_name.split("_")
        record = int(values[1])  # Start from the old record from the loaded file
        iteration = int(values[2]) + 1  # Number of retraining done
        game = ReinforcementTrainingGame(reward_dead=-10)  # Lower death penalty
    else:
        agent = Agent(gamma=0.9, decay=0.99)  # Else training from a fresh agent
        game = ReinforcementTrainingGame()

    # Initialise the run
    game.start_run()
    for i in tqdm(range(args.episodes)):
        done = False
        move_nbr = 0
        while not done:  # While the game is not lost
            # Observe the actual state
            state_old = agent.get_state(game)

            # Ask the agent to choose an action depending on the state
            final_move = agent.act_train(state_old)
            agent.direction = final_move  # Remember the last move (avoid random turnover)

            # Ask the game to process the move and get back the reward, the score and if the snake is dead
            reward, done, score = game.next_tick(final_move)

            # Observe the new state
            state_new = agent.get_state(game)

            # Train the agent over this single step
            agent.training_montage(state_old, reward, state_new, done)

            # Remember this action and its consequence for later
            agent.fill_memory(state_old, reward, state_new, done)

            move_nbr += 1

        # score_distibution[score] += 1
        # Reset the game state
        game.start_run()
        agent.n_games += 1

        if i % 200 == 0:  # Backup weights every 200 runs
            agent.model.save(f"try12/{args.weights}_{record}_{iteration}_{i}.h5")

        # Train the agent based on a random set of moves done recently (1,000,000 move registered)
        agent.train_long_memory()

        if score > record:  # Backup weights if the highest score is beaten
            record = score
            agent.model.save(f"try12/{args.weights}_{record}_{iteration}.h5")

        print('Game', agent.n_games, 'Score', score, 'Record', record, 'Move number', move_nbr)
        total_score += score
        mean_score = total_score / agent.n_games

    # print(score_distibution)
    print(mean_score)
    # One last save of the weights
    agent.model.save(f"try12/{args.weights}_{record}_{iteration}_end.h5")


if __name__ == "__main__":
    training()
