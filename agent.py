import torch
import random
import numpy as np
from collections import deque

from main import SnakeAI
from main import Direction

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = None
        self.trainer = None

    def get_state(self, game):
        head = game.snake[0]

        dir_l = game.velocity == Direction.LEFT
        dir_r = game.velocity == Direction.RIGHT
        dir_u = game.velocity == Direction.UP
        dir_d = game.velocity == Direction.DOWN

        state = [
            dir_l and game.check_collisions([head[0] - 1, head[1]]),  # danger left
            dir_r and game.check_collisions([head[0] + 1, head[1]]),  # danger right
            dir_u and game.check_collisions([head[0], head[1] - 1]),  # danger up
            dir_d and game.check_collisions([head[0], head[1] + 1]),  # danger down

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food_pos[0] < head[0],  # food left
            game.food_pos[0] > head[0],  # food right
            game.food_pos[1] < head[1],  # food up
            game.food_pos[1] > head[1],  # food down

        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # pop left if MAX_MEMORY

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuple
        else:
            mini_sample = self.memory
        #  states, actions, rewards, next_states, dones = zip(*mini_sample)
        for states, actions, rewards, next_states, dones in mini_sample:
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
            print(f"Game:{agent.n_games}, Score:{score}, Record:{record} ")


if __name__ == "__main__":
    train()
