import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from agent import Agent, AgentTrainingGame
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
    agent = Agent(input_size=20)
    game = AgentTrainingGame(agent)
    saving_weights_each_steps = 1000
    print("\n >>> Begin Epsilon = " + str(agent.epsilon))
    print(" >>> Decay = " + str(agent.decay))


    # state = [ grid, score, alive, snake ]

    # Episode LOOP
    for i in tqdm(range(args.episodes)):
        # Game and board reset
        running = True
        print(f"Start game {i}")
        game.start_run()
        # previous_state = agent.get_state_properties(game) old

        while running:
            # Performs the next training action
            # actual_state = game.next_tick()
            infos = game.next_tick()

            # done = actual_state[2]  # Snake alive or not
            # running = game.is_alive()
            # reward = actual_state[1]  # reward = new_score
            # Saves the move in memory
            # agent.fill_memory(previous_state, actual_state, reward, running) Old
            running = game.is_alive()
            # agent.fill_memory(previous_state, actual_state, reward, running)
            agent.fill_memory(infos)

            # Resets iteration for the next move
            # previous_state = actual_state
        print(f"Game {i} over with score: {game.score}")
        # train the weights of the NN after the episode
        agent.training_montage()

        if i % saving_weights_each_steps == 0:
            agent.save(f"weights_temp_{i}.h5")



    agent.save(f"{args.weights}.h5")

    print("\n >>> End Epsilon = " + str(agent.epsilon))


if __name__ == "__main__":
    training()
