import random
from collections import deque

import numpy as np
import tensorflow.python.ops.ragged.ragged_factory_ops
from numpy import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import disable_interactive_logging
from tensorflow import constant
from tensorflow import convert_to_tensor

from gameModule import GUISnakeGame, SnakeGame
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
        disable_interactive_logging()
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

    def act_train(self, actions):
        if random.uniform(0, 1) < self.epsilon:
            if len(actions) == 0:
                print("No future states:", actions)
                return None

            return random.choice(actions)

        best_score = None
        best_state = None
        # scores = self._predict_scores([state for action, state in states]) old
        scores = self._predict_scores(actions)
        # for i, (action, state) in enumerate(states):
        for i, action in enumerate(actions):
            score = scores[i]
            if not best_score or score > best_score:
                best_score = score
                best_state = action

        return best_state

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
        pre_inputs = []
        for state in states:
            to_add = []
            for elem in state:
                if type(elem) == list or type(elem) == tuple:
                    for i in elem:
                        to_add.append(i)
                else:
                    to_add.append(elem)
            pre_inputs.append(to_add)

        input = np.array(pre_inputs, dtype=object).astype('int32')
        # input = [[int(value) if isinstance(value, bool) else value for value in sublist] for sublist in input]

        tensorInput = convert_to_tensor(input)
        predictions = self.model.predict(tensorInput)

        return [prediction[0] for prediction in predictions]

    # def fill_memory(self, previous_state, next_state, reward, done):
    #     self.memory.append((previous_state, next_state, reward, done))

    # def fill_memory(self, previous_state, action, reward, next_state, running):
    #     self.memory.append((previous_state, action, reward, next_state, running))

    def fill_memory(self, info):
        pre_inputs = []
        for elem in info:
            if type(elem) == list or type(elem) == tuple:
                for i in elem:
                    pre_inputs.append(i)
            else:
                pre_inputs.append(elem)
        self.memory.append(pre_inputs)

    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)

    def training_montage(self, batch_size=64, epochs=1):
        if len(self.memory) < batch_size:
            return
        experiences = random.sample(self.memory, batch_size)
        # next_states = [experience[1] for experience in experiences]
        scores = self._predict_scores(experiences)
        dataset = []
        target = []
        for i in range(batch_size):
            experience = experiences[i]
            previous_state = experience[:8]
            action = experience[8:10]
            reward = experience[10]
            _ = experience[10:18]
            running = experience[19]
            if running:
                next_q = self.gamma * scores[i] + reward
            else:
                next_q = reward
            # dataset.append(previous_state)
            dataset.append(experience)
            target.append(next_q)
        self.model.fit(
            dataset, target, batch_size, epochs, verbose=0
        )
        self.epsilon = max(
            self.epsilon * self.decay, self.epsilon_min
        )

    def choose_next_move(self, game: SnakeGame):
        # states = self.get_possible_states(state)
        actions = self.get_actions(game)
        return self.act_train(actions)[1]  # TODO: Remplacer par act_best

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
                # states.append((action, (int(alive), dist_to_food[0], dist_to_food[1], coilness)))
                states.append((action, (dist_to_food[0], dist_to_food[1], coilness)))
        if len(states) == 0:
            # states.append((RIGHT, (int(False), self.foodCloseness(grid, head)[0], self.foodCloseness(grid, head)[1], self.computeCoilness(grid, state[3]))))
            states.append((RIGHT, (self.foodCloseness(grid, head)[0], self.foodCloseness(grid, head)[1],
                                   self.computeCoilness(grid, state[3]))))
        return states

    def get_state_properties(self, game: SnakeGame):  # old
        grid, score, alive, snake = game.get_state()

        dist_to_food = self.foodCloseness(grid, snake[0])
        coilness = self.computeCoilness(grid, snake)

        # return alive, dist_to_food[0], dist_to_food[1], coilness
        return dist_to_food[0], dist_to_food[1]

    def get_state(self, grid, head):
        """
        :param head:
        :param grid:
        :return: [dangerDOWN, dangerUP, dangerRIGHT, dangerLEFT, foodDOWN, foodUP, foodRIGHT, foodLEFT]
        """
        state = [int(not isInGrid(len(grid), head, direction) or
                     grid[head[0] + direction[0]][head[1] + direction[1]] == WALL_CHAR or
                     grid[head[0] + direction[0]][head[1] + direction[1]] == SNAKE_CHAR)
                 for direction in [DOWN, UP, RIGHT, LEFT]]

        # Find the position of the food
        food_pos = (0, 0)
        for i, row in enumerate(grid):
            for j, content in enumerate(row):
                if content == FOOD_CHAR:
                    food_pos = (i, j)

        state.append(int(food_pos[0] > head[0]))
        state.append(int(food_pos[0] < head[0]))
        state.append(int(food_pos[1] > head[1]))
        state.append(int(food_pos[1] < head[1]))

        return state

    def get_actions(self, game: SnakeGame):
        grid = game.grid
        head = game.snake[0]
        current_state = self.get_state(grid, head)
        possible_actions = []

        for action in [DOWN, UP, RIGHT, LEFT]:
            if isInGrid(len(grid), head, action):
                new_grid, new_snake, foodEaten, alive = self.store(grid, game.snake, action, game)
                reward = 10 if foodEaten else -10 if not alive else 0
                running = int(alive)
                new_state = self.get_state(new_grid, new_snake[0])

                possible_actions.append([current_state, action, reward, new_state, running])

        return possible_actions

    def store(self, grid, snake, action, game):
        """
        Make a deep copy of the board and act the action given
        :param snake:
        :param grid:
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
            ):
                alive = False
            else:
                snake_cpy.insert(0, new_pos)
                grid_cpy[new_pos[0]][new_pos[1]] = SNAKE_CHAR
                if grid[new_pos[0]][new_pos[1]] == FOOD_CHAR:
                    foodEaten = True
                    random_cell = game.get_random_cell()
                    grid_cpy[random_cell[0]][random_cell[1]] = FOOD_CHAR
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


class AgentTrainingGame(SnakeGame):

    def __init__(self, learning_agent: Agent):
        super().__init__()
        self.learning_agent = learning_agent
        self.score = 0

    def next_tick(self):
        if self.is_alive():
            actions = self.learning_agent.get_actions(self)
            infos = self.learning_agent.act_train(actions)
            self.set_next_move(infos[1])

            if self.foodEaten:
                self.learning_agent.eat()
            self.move_snake()
            # return self.learning_agent.get_state_properties(self) old
            return infos

        # return self.learning_agent.get_state_properties(self) old
        return self.learning_agent.get_state(self.grid, self.snake[0])


def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = Agent(input_size=4)
    while game.is_running():
        game.next_tick(agent)
        # restart when dead
        while not (game.is_alive()):
            game.start_run()

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
