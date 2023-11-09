from enum import Enum

import pygame
import sys
from random import choice
import numpy as np

pygame.init()


# reset
# rewards
# play(action) => direction
# game_iteration
# is_collisions

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeAI:
    def __init__(self):
        self.NB_CASES = 35
        self.SIZE = 20
        self.WIDTH = self.NB_CASES * self.SIZE
        self.HEIGHT = self.NB_CASES * self.SIZE

        self.RIGHT = [1, 0]
        self.LEFT = [-1, 0]
        self.DOWN = [0, 1]
        self.UP = [0, -1]

        self.display_surface = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Snake")
        self.FramePerSec = pygame.time.Clock()

        self.snake = None
        self.velocity = None
        self.food = None
        self.food_pos = None
        self.score = None
        self.frame_iteration = None

        self.reset_game()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.update_velocity(action)
        self.update_screen()
        self.update_snake()
        self.check_eat()
        self.update_food()

        reward = 0
        game_over = False

        # check collisions
        if self.check_collisions(self.snake[0]) or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        else:
            self.draw_snake()

        # check eat
        if self.check_eat():
            self.food = False
            self.score += 1
            self.add_part()
            reward = 10
            print(self.score)

        self.FramePerSec.tick(10)
        pygame.display.flip()

        return reward, game_over, self.score

    def reset_game(self):
        self.snake = [[2, 5], [2, 4], [2, 3]]
        self.velocity = Direction.RIGHT

        self.food = True
        self.food_pos = self.get_pos_food()

        self.score = 0
        self.frame_iteration = 0

    def update_screen(self):
        self.display_surface.fill((0, 0, 0))

    def update_velocity(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.velocity)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.velocity = new_dir

    def check_collisions(self, head):
        for i in range(1, len(self.snake) - 1):
            if head[0] == self.snake[i][0] and head[1] == self.snake[i][1]:
                return True

        if head[0] < 0 or head[0] > self.NB_CASES - 1:
            return True
        if head[1] < 0 or head[1] > self.NB_CASES - 1:
            return True

    def update_snake(self):
        self.right_shift()
        if self.velocity == Direction.RIGHT:
            self.snake[0][0] += 1
        elif self.velocity == Direction.LEFT:
            self.snake[0][0] -= 1
        elif self.velocity == Direction.DOWN:
            self.snake[0][1] += 1
        elif self.velocity == Direction.UP:
            self.snake[0][1] -= 1


    def draw_snake(self):
        for part in range(len(self.snake) - 1):
            pos_x = self.snake[part][0] * self.SIZE
            pos_y = self.snake[part][1] * self.SIZE
            rect = pygame.Rect(pos_x, pos_y, self.SIZE, self.SIZE)
            pygame.draw.rect(self.display_surface, (0, 0, 255), rect)

    def right_shift(self):
        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = list(self.snake[i - 1])

    def check_eat(self):
        if self.snake[0][0] == self.food_pos[0] and self.snake[0][1] == self.food_pos[1]:
            return True

    def get_pos_food(self):
        all_cases = [(i, j) for i in range(self.NB_CASES) for j in range(self.NB_CASES)]
        free_cases = [case for case in all_cases if case not in self.snake]
        return choice(free_cases)

    def add_part(self):
        last_x = self.snake[len(self.snake) - 1][0]
        last_y = self.snake[len(self.snake) - 1][1]
        self.snake.append([last_x, last_y])

    def update_food(self):
        if not self.food:
            self.food_pos = self.get_pos_food()
            self.food = True
        self.draw_food()

    def draw_food(self):
        rect = pygame.Rect(self.food_pos[0] * self.SIZE, self.food_pos[1] * self.SIZE, self.SIZE,
                           self.SIZE)
        pygame.draw.rect(self.display_surface, (255, 0, 0), rect)


if __name__ == "__main__":
    game = SnakeAI()
    actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    while True:
        action = choice(actions)
        print(action)
        reward, game_over, score = game.play_step(action)
        if game_over:
            break
    print(f"Score finale: {score}, reward: {reward}")

