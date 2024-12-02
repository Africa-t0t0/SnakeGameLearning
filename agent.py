import torch
import random
import numpy as np

from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helpers import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

LR = 0.001

class Agent(object):
    def __init__(self):
        self._number_of_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft
        self.model = Linear_QNet(input_size=11,
                                 hidden_size=256,
                                 output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # TODO: model, trainer

    def get_state(self, game) -> np.array:
        # this is checking if the snake is in danger.
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction, only ONE of them is true.
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # pop left if max_memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # returns a list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        This function only trains for one step!
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # the more games we have, the smaller our epsilon will be
        self.epsilon = 80 - self._number_of_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

            final_move[move] = 1

        return final_move


def train():
    plot_scores = list()
    plot_mean_scores = list()
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # training loop
    while True:
        # get old state
        state_old = agent.get_state(game=game)

        # get move
        final_move = agent.get_action(state=state_old)

        # perform move and get new state
        reward, done, score = game.play_step(action=final_move)
        state_new = agent.get_state(game=game)

        # train short memory
        agent.train_short_memory(state=state_old,
                                 action=final_move,
                                 reward=reward,
                                 next_state=state_new,
                                 done=done)

        agent.remember(state=state_old,
                                 action=final_move,
                                 reward=reward,
                                 next_state=state_new,
                                 done=done)

        if done:
            # train the long memory and plot the result
            game.reset()
            agent._number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            print("Game", agent._number_of_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total_score += score

            mean_score = total_score / agent._number_of_games
            plot_mean_scores.append(mean_score)

            plot(scores=plot_scores, mean_scores=plot_mean_scores)


if __name__ == "__main__":
    train()