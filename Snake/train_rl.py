import argparse
import sys
from pathlib import Path

from IPython import display
from tqdm import tqdm
import matplotlib.pyplot as plt

from agent import Agent, ReinforcementTrainingGame
from gameModule import SnakeGame


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
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    score = 0
    iteration = 0
    if args.retrain:
        agent = Agent(epsilon=0.005, epsilon_min=0.005)
        agent.load(args.retrain)
        file_name = args.retrain.split(".")[0]
        values = file_name.split("_")
        print(values)
        record = int(values[1])
        iteration = int(values[2])+1
        game = ReinforcementTrainingGame(reward_live=0.01, reward_eat=10, reward_dead=-100)
    else:
        agent = Agent()
        game = ReinforcementTrainingGame()
    game.start_run()
    for i in tqdm(range(args.episodes)):
        done = False
        move_nbr = 0
        while not done:
            state_old = agent.get_state(game)

            final_move = agent.act_train(state_old)

            reward, done, score = game.next_tick(final_move)

            state_new = agent.get_state(game)

            if not args.retrain or done: agent.training_montage(state_old, final_move, reward, state_new, done)

            agent.fill_memory(state_old, final_move, reward, state_new, done)

            move_nbr += 1

        game.start_run()
        agent.n_games += 1

        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save(f"try2/{args.weights}_{record}_{iteration}.h5")

        print('Game', agent.n_games, 'Score', score, 'Record', record, 'Move number', move_nbr)
        #
        # plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        # plot_mean_score.append(mean_score)
        # plot(plot_scores, plot_mean_score)

    print(mean_score)
    agent.model.save(f"try2/{args.weights}_{record}_{iteration}.h10")



def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

if __name__ == "__main__":
    training()