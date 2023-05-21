import random
from collections import deque
import pygame

import numpy as np
from numpy import sqrt
import tensorflow as tf

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
            input_size=8,
            epsilon=0.9,
            decay=0.99,
            gamma=0.9,
            loss_fct="mse",
            opt_fct="adam",
            mem=1000,
            metrics=None,
            epsilon_min=0.01
    ):
        tf.keras.utils.disable_interactive_logging()
        self.n_games = 0

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

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(input_size,)))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(4, activation="linear"))
        self.model.compile(
            optimizer=self.opt_fct, loss=self.loss_fct, metrics=self.metrics
        )

    def get_state(self, game: SnakeGame):
        head = game.snake[0]

        state = [
            # Danger Around head
            game.is_collision((head[0] + RIGHT[0], head[1] + RIGHT[1])),
            game.is_collision((head[0] + DOWN[0], head[1] + DOWN[1])),
            game.is_collision((head[0] + LEFT[0], head[1] + LEFT[1])),
            game.is_collision((head[0] + UP[0], head[1] + UP[1])),

            # Food location
            game.food[0] < head[0],  # food left
            game.food[0] > head[0],  # food right
            game.food[1] < head[1],  # food up
            game.food[1] > head[1]  # food down
        ]

        return np.array(state, dtype=int)

    def train_long_memory(self, batch_size=64):
        if len(self.memory) > batch_size:
            sample = random.sample(self.memory, batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.training_montage(states, actions, rewards, next_states, dones)
        self.epsilon = max(
            self.epsilon * self.decay, self.epsilon_min
        )
        print(self.epsilon)

    def training_montage(self, state, action, reward, next_state, done, epochs=1):
        # if len(self.memory) < 64:
        #     return
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            next_state = tf.expand_dims(next_state, 0)
            done = (done,)

        scores = self._predict_scores(next_state)  # Q values with current state
        # print(scores.shape, len(done), scores)
        dataset = []
        target = []
        for i in range(len(done)):
            next_q = self.gamma * scores[i] + reward[i]
            # print(scores[i], reward[i], scores[i]+reward[i])
            # if not done[i]:
            #     next_q = self.gamma * scores[i] + reward[i]
            #     print(scores[i], reward[i], scores[i]+reward[i])
            # else:
            #     next_q = reward[i]
            #     print(next_q)

            # print(state[i], next_q)
            dataset.append(list(state[i]))
            target.append(np.array(next_q))
        dataset = tf.convert_to_tensor(dataset)
        # print(target)
        target = tf.convert_to_tensor(target)
        # print("Shapes: ", dataset.shape, target.shape, len(done))
        # print("Dataset: ", list(dataset[0]), dataset)
        # print("target: ", np.array(target[0]), target)
        self.model.fit(
            dataset, target, len(done), epochs, verbose=0
        )

    def act_train(self, state):
        # eps = 80 - self.n_games
        # if random.randint(0,200) < eps:
        #     return random.choice([RIGHT, LEFT, UP, DOWN])
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([RIGHT, LEFT, UP, DOWN])
        else:
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            prediction = self._predict_scores(state0)
            # print(prediction)
            # print(tf.math.argmax(prediction[0]))
            move = int(tf.math.argmax(prediction[0]))
            final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def act_best(self, state):

        state0 = tf.convert_to_tensor(state, dtype=tf.float32)
        prediction = self._predict_scores(state0)
        move = int(tf.math.argmax(prediction[0]))
        final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def _predict_scores(self, states):
        input = tf.cast(tf.constant(states), dtype=tf.float32)
        # print(input)
        if input.ndim ==1:
            input = tf.expand_dims(input, axis=0)
        # print(input)
        # input = np.array(states)
        predictions = self.model.predict(input)
        # print(predictions)
        return predictions
        # return [prediction[0] for prediction in predictions]

    def fill_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)

    def choose_next_move(self, game):
        state = self.get_state(game)
        return self.act_best(state)  # Remplacer par act_best

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
        if action is not None:
            head = snake[0]
            new_pos = (head[0] + action[0], head[1] + action[1])
            if not (0 <= new_pos[0] < 20  # Correspont à largeur de la grille
                    and 0 <= new_pos[1] < 20  # Correspond à hauteur de la grille
                    and grid[new_pos[0]][new_pos[1]] in [EMPTY_CHAR, FOOD_CHAR]
            ):  # Todo: trouver une solution pour que les "20" soit variable
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
                if grid[snake_part[0] + direction[0]][snake_part[1] + direction[1]] == SNAKE_CHAR:
                    coilness += 1
        return coilness


class ReinforcementTrainingGame(GUISnakeGame):
    def __init__(self, reward_live=0.1, reward_eat=10, reward_dead=-10):
        super().__init__()
        # self.init_pygame()
        self.mps = 50
        self.reward_live = reward_live
        self.reward_eat = reward_eat
        self.reward_dead = reward_dead

    def next_tick(self, action):
        # for event in pygame.event.get():
        #     pass
        reward = 0
        self.next_move = action
        if self.is_alive():
            reward = self.reward_live
            self.move_snake()
            if self.foodEaten:
                reward = self.reward_eat
            elif not self.is_alive():
                reward = self.reward_dead
        # self.draw()
        # self.clock.tick(30)
        # self.frame += 1
        reward = [reward if pos_act == action else 0 for pos_act in [RIGHT, LEFT, UP, DOWN] ]
        return reward, not self.is_alive(), self.score

def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = Agent(input_size=4)

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
