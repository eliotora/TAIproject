import random
from collections import deque

import numpy as np
from numpy import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import constant
from tensorflow import convert_to_tensor

from gameModule import GUISnakeGame
from gameModule import (
    RIGHT,
    LEFT,
    DOWN,
    UP,
    SNAKE_CHAR,
    EMPTY_CHAR,
    FOOD_CHAR,
    WALL_CHAR,
)


class Agent:
    """
    The Deep Q-learning agent
    It will interact with the Snake environment
    """

    def __init__(
            self,
            input_size=4,
            epsilon=0.9,
            decay=0.9995,
            gamma=0.95,
            loss_fct="mse",
            opt_fct="adam",
            mem=2000,
            metrics=None,
            epsilon_min=0.1
    ):
        if metrics is None:
            metrics = ["mean_squared_error"]
        self.input_size = input_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.loss_fct = loss_fct
        self.opt_fct = opt_fct
        self.memory = deque(maxlen=mem)
        self.decay = decay
        self.metrics = metrics
        self.moves = []
        self.epsilon_min = epsilon_min

        self.model = Sequential()
        self.model.add(Dense(64, activation="relu", input_shape=(input_size,)))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(32, activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(
            optimizer=self.opt_fct, loss=self.loss_fct, metrics=self.metrics
        )

    def act_train(self, states):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(states)[0]

        best_score = None
        best_state = None

        scores = self._predict_scores([state for action, state in states])
        for i, (action, state) in enumerate(states):
            score = scores[i]
            if not best_score or score > best_score:
                best_score = score
                best_state = (action, state)

        print(scores)
        return best_state[0]

    def act_best(self, states):
        best_score = None
        best_state = None

        scores = self._predict_scores([state for action, state in states])
        for i, (action, state) in enumerate(states):
            score = scores[i]
            if not best_score or score > best_score:
                best_score = score
                best_state = (action, state)

        return best_state[0]

    def _predict_scores(self, states):
        input = np.array(states)
        input = [[int(value) if isinstance(value, bool) else value for value in sublist] for sublist in input]

        newInput=[]
        # Split the tuples within the list
        for i in range(len(input)):
            temp=[]
            for elem in input[i]:
                if isinstance(elem, int):
                    temp.append(elem)
                else:
                    for val in elem:
                        temp.append(val)
            newInput.append(temp)

        tensorInput = convert_to_tensor(newInput)
        predictions = self.model.predict(tensorInput)

        return [prediction[0] for prediction in predictions]

    def fill_memory(self, previous_state, next_state, reward, done):
        self.memory.append((previous_state, next_state, reward, done))

    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)

    def training_montage(self, batch_size=64, epochs=1):
        if len(self.memory) < batch_size:
            return
        experiences = random.sample(self.memory, batch_size)

        next_states = [experience[1] for experience in experiences]
        scores = self._predict_scores(next_states)
        dataset = []
        target = []
        for i in range(batch_size):
            previous_state, _, reward, done = experiences[i]
            if not done:
                next_q = self.gamma * scores[i] + reward
            else:
                next_q = reward
            dataset.append(previous_state)
            target.append(next_q)
        self.model.fit(
            dataset, target, batch_size, epochs, verbose=0
        )
        self.epsilon = max(
            self.epsilon * self.decay, self.epsilon_min
        )

    def choose_next_move(self, state):
        states = self.get_possible_states(state)
        return self.act_train(states)  # Remplacer par act_best

    def get_possible_states(self, state):
        """
        state = [grid, score, alive, snake]
        Choisir des features pour donner une idée à l'agent de son "avancée"
        Ex: alive (bool), distance de la bouffe (heur), enroulement du serpent, faim

        Enroulement du serpent: nombre de bout du serpent qui est à côté d'un autre bout de serpent
        -> Augmente les chances de survie
        == Coilness

        Faim: étapes depuis la dernière nourriture mangée -> Encourage à se déplacer vers la bouffe

        :return: Une liste de tuples (action, (features))
        """
        states = []
        grid = state[0]
        head = state[3][0]
        for action in [RIGHT, DOWN, LEFT, UP]:
            if isInGrid(len(grid), head, action):
                next_position = grid[head[0] + action[0]][head[1] + action[1]]
                if next_position == WALL_CHAR and next_position == SNAKE_CHAR:
                    pass
                new_grid, new_snake, foodEaten, alive = self.store(grid, state[3], action)
                dist_to_food = self.foodCloseness(grid, new_snake[0])
                coilness = self.computeCoilness(new_grid, new_snake)
                # TODO: Implement hunger
                states.append((action, [alive, dist_to_food, coilness]))
        return states

    def store(self, grid, snake, action):
        """
        Make a deep copy of the board and act the action given
        :param action: The action to be taken
        :return: the new grid
        """
        grid_cpy = [x[:] for x in grid]
        snake_cpy = [x[:] for x in snake]
        foodEaten = False
        alive = True
        size = len(grid)

        if action is not None:
            head = snake[0]
            new_pos = (head[0] + action[0], head[1] + action[1])
            if not (0 <= new_pos[0] < size  # Correspont à largeur de la grille
                    and 0 <= new_pos[1] < size  # Correspond à hauteur de la grille
                    and grid[new_pos[0]][new_pos[1]] in [EMPTY_CHAR, FOOD_CHAR]
            ):  # Todo: trouver une solution pour que les "20" soit variable  : done
                alive = False
            else:
                snake_cpy.insert(0, new_pos)
                grid_cpy[new_pos[0]][new_pos[1]] = SNAKE_CHAR
                if grid[new_pos[0]][new_pos[1]] == FOOD_CHAR:
                    foodEaten = True
                else:
                    tail = snake_cpy.pop()
                    grid_cpy[tail[0]][tail[1]] = EMPTY_CHAR

        return grid_cpy, snake_cpy, foodEaten, alive

    def reset_state(self):
        return

    def eat(self):  # TODO
        return

    def foodCloseness(self, grid, head):
        """
        Compute strait line distance to food
        """
        food_pos = None
        for i, row in enumerate(grid):
            for j, content in enumerate(row):
                if content == FOOD_CHAR:
                    food_pos = (i, j)
        return sqrt((food_pos[0] - head[0]) ** 2), sqrt((food_pos[1] - head[1]) ** 2)

    def computeCoilness(self, grid, snake):
        """
        Compute how much piece of snake surround each piece of snake
        """
        coilness = 0
        for snake_part in snake:
            for direction in [RIGHT, LEFT, UP, DOWN]:
                if isInGrid(len(grid), snake_part, direction):
                    if grid[snake_part[0] + direction[0]][snake_part[1] + direction[1]] == SNAKE_CHAR:
                        coilness += 1
        return coilness


def isInGrid(lenGrid, snake_part, direction):
    if snake_part[0] + direction[0] != lenGrid and snake_part[0] + direction[0] != 0 \
            and snake_part[1] + direction[1] != lenGrid and snake_part[1] + direction[1] != 0:
        return True

    return False


def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = Agent(input_size=4)

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
