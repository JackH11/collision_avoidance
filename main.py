import pygame
import sys
import math
import uuid

from nn.nn import nll_gaussian, ClippedLogVar
from utils import get_model, save_data, get_data
from model_prediction import agent_uncertain_predict, update_item_async, make_nn_prediction, make_simple_prediction
from gym_env import MovingItem, MovingAgent
import numpy as np
from config import CONFIG
from stable_baselines3 import DQN
from concurrent.futures import ThreadPoolExecutor
from gym_env import MovingAvoidanceEnv


def draw_item_async(item, surface,color=None):
    """Asynchronous draw function"""

    item.draw(surface,color)
    return item

data = []

SAVE = False
RENDER = True

# Screen dimensions
WIDTH, HEIGHT = 400, 400
ITEM_RADIUS = 5
ITEM_COUNT = 5
MAX_STEPS = 1000
AGENT_SPEED = 3.0

WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
RED = tuple(CONFIG["colors"]["red"])
BLUE = tuple(CONFIG["colors"]["blue"])
GREEN = tuple(CONFIG["colors"]["green"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])
ORANGE = tuple(CONFIG["colors"]["orange"])

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
    #agent = MovingAgent(make_nn_prediction,add_noise=False)
    agent = MovingAgent(make_simple_prediction, add_noise=False)

    dqn_model = DQN.load("agents/dqn_avoidance_agent4")

    running = True
    frame = 0
    episode = uuid.uuid4()
    """

    df = get_data("train_lag")
    df_sorted = df.sort_values(by=["frame", "item"], ascending=[True, True])

    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted["frame_change"] = df_sorted["frame"] != df_sorted["frame"].shift()

    for idx, row in df_sorted.iterrows():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if row["frame_change"]:
            pygame.time.wait(100)
            screen.fill(WHITE)      

        item = MovingItem(add_noise=False)
        item.x = row["x_lag_20"]
        item.y = row["y_lag_20"]
        item.draw(screen)
        pygame.draw.circle(screen, (255,0,0), (row["x"],row["y"]), 5, width=0)
        pygame.display.flip()
        
       #clock.tick(30)  # Increased from 60 to 120 FPS for smoother motion
    """

    while frame < 10000:
        print("Frame: ", frame)

        # Update All Items
        update_futures = [executor.submit(update_item_async, obstacle) for obstacle in obstacles]

        # Wait for all updates to complete
        for future in update_futures:
            future.result()

        if SAVE:
            for obstacle in obstacles:
                data.append({'episode':episode,'frame':frame,'item':obstacle.id, 'x':obstacle.x, 'y':obstacle.y, 'vx':obstacle.vx, 'vy':obstacle.vy})
        

        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(WHITE)

            draw_futures = []
            for obstacle in obstacles:
                draw_futures.append(executor.submit(draw_item_async, obstacle, screen, LIGHT_GREY))

            # Wait for all draws to complete
            for future in draw_futures:
                future.result()

            # Run predictions
            uncertainty_predictions = agent.make_predictions(obstacles)

            for obstacle in obstacles:
                obstacle.draw(screen, obstacle)

            obs = agent.get_observation(obstacles)

            action, _states = dqn_model.predict(obs)

            dx, dy = env._action_to_velocity(action)
            agent.vx = dx
            agent.vy = dy

            agent.update()

            agent.draw(screen,obstacles,uncertainty_predictions)

            env.draw_predictions(screen,obstacles,uncertainty_predictions)

        if RENDER:
            pygame.display.flip()
            clock.tick(120)  # Increased from 60 to 120 FPS for smoother motion

        frame += 1

    
    if SAVE:
        # Check if file exists to determine if we need to write header
        save_data(data, fieldnames=['episode', 'frame','item', 'x', 'y', 'vx', 'vy'])
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 