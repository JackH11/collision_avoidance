import pygame
import sys
import uuid

from utils import get_model, save_data, get_data
from model_prediction import agent_uncertain_predict, update_item_async, make_nn_prediction, make_simple_prediction
from gym_env import MovingItem, MovingAgent
import numpy as np
from config import CONFIG
from stable_baselines3 import DQN
from concurrent.futures import ThreadPoolExecutor
from gym_env import MovingAvoidanceEnv
from functools import partial




data = []

SAVE = False
RENDER = True

# Screen dimensions
WIDTH = CONFIG["window"]["width"]
HEIGHT = CONFIG["window"]["height"]
ITEM_RADIUS = CONFIG["obstacle"]["radius"]
ITEM_COUNT = CONFIG["obstacle"]["count"]
MAX_STEPS = 1000
AGENT_SPEED = CONFIG["agent"]["speed"]

WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
RED = tuple(CONFIG["colors"]["red"])
BLUE = tuple(CONFIG["colors"]["blue"])
GREEN = tuple(CONFIG["colors"]["green"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])
ORANGE = tuple(CONFIG["colors"]["orange"])
PURPLE = tuple(CONFIG["colors"]["purple"])

if RENDER:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('2D Moving Items - Prediction Agent')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)  # default font, size 30

env = MovingAvoidanceEnv()

def render(items):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pass

executor = ThreadPoolExecutor(max_workers=4)

def main():
    obstacles = [MovingItem(add_noise=True) for i in range(ITEM_COUNT)]

    # make_simple_prediction and make_nn_prediction
    agent = MovingAgent(make_simple_prediction, add_noise=False)


    dqn_model = DQN.load("dqn_avoidance_agent5")

    running = True
    frame = 0
    episode = uuid.uuid4()

    while frame < 10000:
        print("Frame: ", frame)

        # Update All Items
        update_futures = [executor.submit(update_item_async, obstacle) for obstacle in obstacles]

        # Wait for all updates to complete
        for future in update_futures:
            future.result()

        if SAVE:
            # Collect Data about Obstacles locations
            for obstacle in obstacles:
                data.append({'episode':episode,'frame':frame,'item':obstacle.id, 'x':obstacle.x, 'y':obstacle.y, 'vx':obstacle.vx, 'vy':obstacle.vy})
        

        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(WHITE)

            # draw obstacles
            draw_func = partial(draw_item_async, surface=screen, color=LIGHT_GREY)
            list(executor.map(draw_func, obstacles))

            # Run predictions
            uncertainty_predictions = agent.make_predictions(obstacles)

            for obstacle in obstacles:
                obstacle.draw(screen, obstacle)

            obs = agent.get_observation(obstacles, env)

            action, _states = dqn_model.predict(obs)

            dx, dy = env._action_to_velocity(action)
            agent.vx = dx
            agent.vy = dy



            reward = env.get_reward(agent)
            text_surface = font.render(f"Reward: {reward}", True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.topright = (WIDTH - 10, 10)  # 10 px padding from the top-right corner
            screen.blit(text_surface, text_rect)

            agent.update()

            env.draw_arrow_from_base(screen, BLACK, agent.x, agent.y, action)
            agent.draw(screen,env,obstacles,uncertainty_predictions,True)

            #env.draw_predictions(screen,obstacles,uncertainty_predictions)
            env.draw_goal(screen)

            pygame.display.flip()
            clock.tick(120)

        frame += 1

    
    if SAVE:
        # Check if file exists to determine if we need to write header
        save_data(data, fieldnames=['episode', 'frame','item', 'x', 'y', 'vx', 'vy'])
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 