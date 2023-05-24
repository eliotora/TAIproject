import random
from collections import deque

import numpy as np
import tensorflow as tf

from gameModule import GUISnakeGame, SnakeGame
from gameModule import (
    RIGHT,
    LEFT,
    DOWN,
    UP,
)


class Agent:
    """
    The Deep Q-learning agent
    It will interact with the Snake environment
    It can be trained using the train_rl.py file or be run by the game given a weight file
    """

    def __init__(
            self,
            input_size=8,
            epsilon=0.9,
            decay=0.9995,
            gamma=0.9,
            loss_fct="mse",
            opt_fct="adam",
            mem=1000000,
            metrics=None,
            epsilon_min=0.1
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
        self.model.add(tf.keras.layers.Dense(64, activation="relu", input_shape=(input_size,)))
        # self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(4, activation="linear"))
        self.model.compile(
            optimizer=self.opt_fct, loss=self.loss_fct, metrics=self.metrics
        )

    def get_state(self, game: SnakeGame):
        """
        This function take the snake game as an input, and compute different information about the
        snake environment, that will be fed to the neural network.
        :param game: snake game object
        :return: information the neural network take as input (representing a state)
        """
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

            # Last movement
            # self.direction[0],
            # self.direction[1]

            # How many snakes bit in each direction
            # len([part for part in game.snake if part[0] == head[0] and part[1] > head[1]]),  # Right [0, 1]
            # len([part for part in game.snake if part[0] == head[0] and part[1] < head[1]]),  # Left [0, -1]
            # len([part for part in game.snake if part[0] < head[0] and part[1] == head[1]]),  # Up [-1, 0]
            # len([part for part in game.snake if part[0] > head[0] and part[1] == head[1]])  # Down [1, 0]

            # Idea: Distance to food: reward if it lowers
        ]
        return np.array(state, dtype=int)

    def train_long_memory(self, batch_size=64):
        """
        This function extract previous instances to upgrade the behavior, and update epsilon
        :param batch_size: Number of instance used the learning
        :return: none
        """
        # Selection of "model" in the memory
        if len(self.memory) > batch_size:
            sample = random.sample(self.memory, batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, dones = zip(*sample)
        self.training_montage(states, rewards, next_states, dones)
        self.epsilon = max(
            self.epsilon * self.decay, self.epsilon_min
        )

    def training_montage(self, state, reward, next_state, done):
        """
        This function allow the learning of our agent using Q-learning and the neural network
        Parameters can be a single value or a batch of values in a numpy array
        :param state: Information of the environment
        :param reward: Reward of the action done
        :param next_state: New state of the snake after an action
        :param done: Boolean about the state of the game (True when game finished)
        :return: None
        """
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state, 0)
            reward = tf.expand_dims(reward, 0)
            next_state = tf.expand_dims(next_state, 0)
            done = (done,)
        # Q values with current state
        scores = self._predict_scores(next_state)  # Prediction with the neural network
        dataset = []
        target = []
        # Apply Q-learning method
        for i in range(len(done)):
            if not done[i]:
                next_q = self.gamma * scores[i] + reward[i]
            else:
                next_q = reward[i]

            dataset.append(list(state[i]))
            target.append(np.array(next_q))

        dataset = tf.convert_to_tensor(dataset)
        target = tf.convert_to_tensor(target)

        self.model.train_on_batch(
            dataset, target
        )

    def act_train(self, state):
        """
        Train the model using random moves and predictions
        :param state: Information about the environment of the snake
        :return: Next move
        """
        if random.uniform(0, 1) < self.epsilon:
            possibilities = [RIGHT, LEFT, UP, DOWN]
            possibilities.remove(self.direction)
            return random.choice(possibilities)
        else:
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            prediction = self._predict_scores(state0)   # Prediction with the neural network
            move = int(tf.math.argmax(prediction[0]))   # Take the best move predicted
            final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def act_best(self, state):
        """
        Use the best possible prediction with  the model (no randomness)
        :param state: Information about the environment of the snake
        :return: next move
        """
        state0 = tf.convert_to_tensor(state, dtype=tf.float32)
        prediction = self._predict_scores(state0)   # Prediction with the neural network
        move = int(tf.math.argmax(prediction[0]))   # Take the best move predicted
        final_move = [RIGHT, LEFT, UP, DOWN][move]

        return final_move

    def _predict_scores(self, states):
        """
        Predict the new action of the snake using the neural network
        :param states: Information about the environment of the snake
        :return: Possible moves with ponderation
        """
        input = tf.cast(tf.constant(states), dtype=tf.float32)
        if input.ndim == 1:
            input = tf.expand_dims(input, axis=0)

        predictions = self.model.predict_on_batch(input)
        # predictions = self.model.predict(input) Causes memory leaks over time
        return predictions

    def fill_memory(self, state, action, reward, next_state, done):
        """
        Fill the buffer with previous experiences
        :param state:original state
        :param action: the action chosen by the network
        :param reward:reward received
        :param next_state:state after the action
        :param done:boolean value to signify whether the end of an episode is reached
        """
        self.memory.append((state, action, reward, next_state, done))

    def save(self, path: str):
        """
        save the weights of the network
        :param path: filepath where weights are saved
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
        load the weights of the network
        :param path: filepath where weights are saved
        """
        self.model.load_weights(path)

    def choose_next_move(self, game):
        """
        Return the move chosen by the agent
        :param game: the game object in order to access the state
        :return: the move chosen in [RIGHT, LEFT, UP, DOWN]
        """
        state = self.get_state(game)
        self.direction = self.act_best(state)
        return self.direction

    def reset_state(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass

    def eat(self):
        """
        This function is useless here, it is a placeholder for a function needed in the other
        algorithm.
        """
        pass


class ReinforcementTrainingGame(SnakeGame):
    """
    Special kind of game used to train the reinforcement learning agent
    """
    def __init__(self, reward_live=0, reward_eat=10, reward_dead=-100):
        super().__init__()
        self.reward_live = reward_live  # Reward given just for live one more step
        self.reward_eat = reward_eat  # Reward given for eating a piece of food
        self.reward_dead = reward_dead  # Reward given when dying

    def next_tick(self, action):
        reward = 0
        self.next_move = action  # Store the action
        if self.is_alive():  # Compute the reward
            reward = self.reward_live
            self.move_snake()  # Move the snake
            if self.foodEaten:
                reward = self.reward_eat
            elif not self.is_alive():
                reward = self.reward_dead
        # Give the reward for the corresponding action
        reward = [reward if pos_act == action else 0 for pos_act in [RIGHT, LEFT, UP, DOWN]]
        return reward, not self.is_alive(), self.score


def main():
    game = GUISnakeGame()
    game.init_pygame()
    agent = Agent()

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()


if __name__ == "__main__":
    main()
