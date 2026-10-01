"""
Interactive Pygame demo — the canonical renderer for this project.

Gymnasium ``env.render()`` is intentionally a stub; all visual debugging
(grid channels, goal, action arrow) happens here.

Loads the Phase 2 best CNN model when present; falls back to the legacy
``dqn_avoidance_agent5`` zip. Override with ``--model`` / ``--algo``.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pygame

from config import CONFIG
from gym_env import MovingAgent, MovingAvoidanceEnv, MovingItem, draw_item_async
from model_loader import load_model
from model_prediction import make_simple_prediction, update_item_async
from utils import save_data

DEFAULT_MODEL_CANDIDATES = [
    # Prefer Phase 2 CNN checkpoints when present locally (gitignored).
    "models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip",
    "models/ppo_CnnPolicy_easy_s0_cont/final_model.zip",
    "models/ppo_CnnPolicy_easy_s0_cont/best_model.zip",
    "models/ppo_CnnPolicy_easy_s0/final_model.zip",
    "models/ppo_CnnPolicy_easy_s0/best_model.zip",
    "models/qrdqn_CnnPolicy_easy_s0/best_model.zip",
    "dqn_avoidance_agent5.zip",
    "dqn_avoidance_agent5",
]


def _pick_default_model() -> str:
    for path in DEFAULT_MODEL_CANDIDATES:
        if Path(path).exists() or Path(f"{path}.zip").exists():
            return path
    return "dqn_avoidance_agent5"


def parse_args():
    p = argparse.ArgumentParser(description="Collision-avoidance Pygame demo")
    p.add_argument(
        "--model",
        default=None,
        help="SB3 zip to load (default: newest Phase 2 best, else dqn_avoidance_agent5)",
    )
    p.add_argument(
        "--algo",
        default=None,
        choices=["ppo", "qrdqn", "dqn"],
        help="Algorithm hint for loading (auto-detected when omitted)",
    )
    p.add_argument(
        "--scenario",
        default=None,
        choices=["easy", "baseline", "hard"],
        help="Override scenario (default: config.yaml)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model or _pick_default_model()

    data = []
    SAVE = False
    RENDER = True

    width = CONFIG["window"]["width"]
    height = CONFIG["window"]["height"]

    white = tuple(CONFIG["colors"]["white"])
    black = tuple(CONFIG["colors"]["black"])
    light_grey = tuple(CONFIG["colors"]["light_grey"])

    if RENDER:
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Moving Items - Prediction Agent")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 30)

    env = MovingAvoidanceEnv(scenario=args.scenario)
    item_count = env.ITEM_COUNT
    executor = ThreadPoolExecutor(max_workers=4)

    policy, algo_name, resolved = load_model(model_path, algo=args.algo)
    print(f"Loaded {algo_name} from {resolved} (scenario={env.scenario_name})")

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
        for _ in range(item_count)
    ]

    agent = MovingAgent(make_simple_prediction, add_noise=False, env_ref=env)

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

            screen.fill(white)

            draw_func = partial(draw_item_async, surface=screen, color=light_grey)
            list(executor.map(draw_func, obstacles))

            uncertainty_predictions = agent.make_predictions(obstacles)

            for obstacle in obstacles:
                obstacle.draw(screen)

            # Sync env.obstacles so get_reward / goal checks stay consistent.
            env.obstacles = obstacles
            env.agent = agent

            obs = agent.get_observation(obstacles, env)
            action, _states = policy.predict(obs, deterministic=True)
            action = int(action)

            dx, dy = env._action_to_velocity(action)
            agent.vx = dx
            agent.vy = dy

            reward = env.get_reward(agent)
            text_surface = font.render(
                f"Reward: {reward:.1f} | {algo_name}", True, black
            )
            text_rect = text_surface.get_rect()
            text_rect.topright = (width - 10, 10)
            screen.blit(text_surface, text_rect)

            agent.update()

            env.draw_arrow_from_base(screen, black, agent.x, agent.y, action)
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
