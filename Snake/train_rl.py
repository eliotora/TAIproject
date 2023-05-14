import math
import sys
import argparse

import numpy as np

from gameModule import GUISnakeGame
import tensorflow as tf
from tensorflow import keras

from snakeTrainer import SnakesManager
from agent import Agent
from tqdm import tqdm
import argparse
from pathlib import Path

FOOD_CHAR = "@"


def get_apple_position(grid):
    """
    Get the position of the apple in the grid
    :param grid: the grid of the game
    :return: the position of the apple
    """
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == FOOD_CHAR:
                return i, j


def angle_with_apple(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position[0]) - np.array(snake_position[0][0])
    snake_direction_vector = np.array(snake_position[0][0]) - np.array(snake_position[0][1])

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector / norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector / norm_of_snake_direction_vector
    angle = math.atan2(
        apple_direction_vector_normalized * snake_direction_vector_normalized - apple_direction_vector_normalized * snake_direction_vector_normalized,
        apple_direction_vector_normalized * snake_direction_vector_normalized + apple_direction_vector_normalized * snake_direction_vector_normalized) / math.pi
    return angle


def training(render=False):
    """
    The training of the agent with the tetris environment
    """
    parser = argparse.ArgumentParser(
        description="The Snake game trainer for RL."
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        help="Path to weights file to save to (default=weights.h5).",
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
        print()
        print(
            f"File {args.weights}, already exists, do you want to overwrite ?"
        )
        y = input("Type yes or no: ")
        if y != "yes":
            print("Aborting.")
            sys.exit()

    # --- Initialisation --- #
    game = GUISnakeGame()
    agent = Agent()

    game.init_pygame()

    while game.is_running():
        game.next_tick()


        grid, score, alive, snake = game.get_state()
        if get_apple_position(grid):
            print(generate_next_direction(snake, angle_with_apple(snake, get_apple_position(grid))))

    game.cleanup_pygame()


def generate_next_direction(snake_position, angle_with_apple):
    direction = 0

    if angle_with_apple > 0:
        direction = 1
    elif angle_with_apple < 0:
        direction = -1
    else:
        direction = 0

    current_direction_vector = np.array(snake_position[0][0]) - np.array(snake_position[0][1])
    left_direction_vector = np.array([current_direction_vector[1], -current_direction_vector[0]])
    right_direction_vector = np.array([-current_direction_vector[1], current_direction_vector[0]])

    new_direction = current_direction_vector
    if direction == -1:
        new_direction = left_direction_vector
    if direction == 1:
        new_direction = right_direction_vector

    button_direction = generate_button_direction(new_direction)

    return direction, button_direction


def generate_button_direction(new_direction):
    button_direction = 0
    if new_direction.tolist() == [10, 0]:
        button_direction = 1
    elif new_direction.tolist() == [-10, 0]:
        button_direction = 0
    elif new_direction.tolist() == [0, 10]:
        button_direction = 2
    else:
        button_direction = 3

    return button_direction


if __name__ == "__main__":
    training()
