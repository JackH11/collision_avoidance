"""
Interactive Pygame demo — the canonical renderer for this project.

Gymnasium ``env.render()`` is intentionally a stub; all visual debugging
(grid channels, goal, action arrow) happens here.
"""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pygame
from stable_baselines3 import DQN

from config import CONFIG
from gym_env import MovingAgent, MovingAvoidanceEnv, MovingItem, draw_item_async
from model_prediction import make_simple_prediction, update_item_async
from utils import save_data

data = []

SAVE = False
RENDER = True

# Screen dimensions
WIDTH = CONFIG["window"]["width"]
HEIGHT = CONFIG["window"]["height"]
ITEM_COUNT = CONFIG["obstacle"]["count"]

WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])

if RENDER:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Moving Items - Prediction Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

# Env supplies goal / reward / action helpers; display is owned by this script.
env = MovingAvoidanceEnv()
executor = ThreadPoolExecutor(max_workers=4)


def main():
    obstacles = [
        MovingItem(
            add_noise=env.obstacle_noise,
            speed_min=env.speed_min,
            speed_max=env.speed_max,
            max_speed=env.obstacle_max_speed,
            noise_scale=env.noise_scale,
            width=env.width,
            height=env.height,
            radius=env.item_radius,
        )
        for _ in range(ITEM_COUNT)
    ]

    agent = MovingAgent(make_simple_prediction, add_noise=False, env_ref=env)
    dqn_model = DQN.load("dqn_avoidance_agent5")

    running = True
    frame = 0
    episode = uuid.uuid4()

    while frame < 10000 and running:
        print("Frame: ", frame)

        update_futures = [
            executor.submit(update_item_async, obstacle) for obstacle in obstacles
        ]
        for future in update_futures:
            future.result()

        if SAVE:
            for obstacle in obstacles:
                data.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "item": obstacle.id,
                        "x": obstacle.x,
                        "y": obstacle.y,
                        "vx": obstacle.vx,
                        "vy": obstacle.vy,
                    }
                )

        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(WHITE)

            draw_func = partial(draw_item_async, surface=screen, color=LIGHT_GREY)
            list(executor.map(draw_func, obstacles))

            uncertainty_predictions = agent.make_predictions(obstacles)

            for obstacle in obstacles:
                obstacle.draw(screen)

            # Sync env.obstacles so get_reward / goal checks stay consistent.
            env.obstacles = obstacles
            env.agent = agent

            obs = agent.get_observation(obstacles, env)
            action, _states = dqn_model.predict(obs)

            dx, dy = env._action_to_velocity(action)
            agent.vx = dx
            agent.vy = dy

            reward = env.get_reward(agent)
            text_surface = font.render(f"Reward: {reward}", True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.topright = (WIDTH - 10, 10)
            screen.blit(text_surface, text_rect)

            agent.update()

            env.draw_arrow_from_base(screen, BLACK, agent.x, agent.y, action)
            agent.draw(screen, env, obstacles, uncertainty_predictions, True)
            env.draw_goal(screen)

            pygame.display.flip()
            clock.tick(120)

        frame += 1

    if SAVE:
        save_data(data, fieldnames=["episode", "frame", "item", "x", "y", "vx", "vy"])

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()