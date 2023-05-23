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
            input_size=12,
            epsilon=0.9,
            decay=0.9995,
            gamma=0.9,
            loss_fct="mse",
            opt_fct="adam",
            mem=1000000,
            metrics=None,
            epsilon_min=0.01
    ):
        tf.keras.utils.disable_interactive_logging()
        self.n_games = 0
        self.direction = RIGHT

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
        self.model.add(tf.keras.layers.Dense(64, activation="linear", input_shape=(input_size,)))
        # self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="linear"))
        self.model.add(tf.keras.layers.Dense(4, activation="linear"))
        self.model.compile(
            optimizer=self.opt_fct, loss=self.loss_fct, metrics=self.metrics
        )

    def get_state(self, game: SnakeGame):
        head = game.snake[0]

        state = [
            # Danger Around head
            game.is_collision((head[0] + RIGHT[0], head[1] + RIGHT[1])),
            game.is_collision((head[0] + LEFT[0], head[1] + LEFT[1])),
            game.is_collision((head[0] + UP[0], head[1] + UP[1])),
            game.is_collision((head[0] + DOWN[0], head[1] + DOWN[1])),

            # Food location
            game.food[1] > head[1],  # food right
            game.food[1] < head[1],  # food left
            game.food[0] < head[0],  # food up
            game.food[0] > head[0],  # food down

            # How many snakes bit in each direction
            len([part for part in game.snake if part[0] == head[0] and part[1] > head[1]]),  # Right [0, 1]
            len([part for part in game.snake if part[0] == head[0] and part[1] < head[1]]),  # Left [0, -1]
            len([part for part in game.snake if part[0] < head[0] and part[1] == head[1]]),  # Up [-1, 0]
            len([part for part in game.snake if part[0] > head[0] and part[1] == head[1]])  # Down [1, 0]

            # Idea: Distance to food: reward if it lowers
        ]

        return np.array(state, dtype=int)

    def train_long_memory(self, batch_size=128):
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
        dataset = []
        target = []
        for i in range(len(done)):
            if not done[i]:
                next_q = self.gamma * scores[i] + reward[i]
                # print(scores[i], reward[i], scores[i]+reward[i])
            else:
                next_q = reward[i]
                # print(next_q)

            # print(state[i], next_q)
            dataset.append(list(state[i]))
            target.append(np.array(next_q))
        dataset = tf.convert_to_tensor(dataset)
        target = tf.convert_to_tensor(target)

        self.model.train_on_batch(
            dataset, target
        )
        # self.model.fit(
        #     dataset, target, len(done), epochs, verbose=0
        # )

    def act_train(self, state):

        if random.uniform(0, 1) < self.epsilon:
            possibilities = [RIGHT, LEFT, UP, DOWN]
            possibilities.remove(self.direction)
            return random.choice(possibilities)
        else:
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            prediction = self._predict_scores(state0)
            move = int(tf.math.argmax(prediction[0]))
            final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def act_best(self, state):

        state0 = tf.convert_to_tensor(state, dtype=tf.float32)
        prediction = self._predict_scores(state0)
        print(prediction)
        move = int(tf.math.argmax(prediction[0]))
        final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def _predict_scores(self, states):
        input = tf.cast(tf.constant(states), dtype=tf.float32)
        if input.ndim ==1:
            input = tf.expand_dims(input, axis=0)

        predictions = self.model.predict_on_batch(input)
        # predictions = self.model.predict(input)
        return predictions

    def fill_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path: str):
        self.model.save_weights(path)

    def load(self, path: str):
        self.model.load_weights(path)

    def choose_next_move(self, game):
        state = self.get_state(game)
        self.direction = self.act_best(state)
        return self.direction

    def reset_state(self):
        return

    def eat(self):
        return


class ReinforcementTrainingGame(SnakeGame):
    def __init__(self, reward_live=0, reward_eat=10, reward_dead=-10):
        super().__init__()
        self.reward_live = reward_live
        self.reward_eat = reward_eat
        self.reward_dead = reward_dead

    def next_tick(self, action):

        reward = 0
        self.next_move = action
        if self.is_alive():
            reward = self.reward_live
            self.move_snake()
            if self.foodEaten:
                reward = self.reward_eat
            elif not self.is_alive():
                reward = self.reward_dead

        reward = [reward if pos_act == action else 0 for pos_act in [RIGHT, LEFT, UP, DOWN]]
        return reward, not self.is_alive(), self.score

def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = Agent(input_size=8)

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
