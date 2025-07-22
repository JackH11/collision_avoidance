import pygame
import sys
import math
import uuid

from nn.nn import nll_gaussian, ClippedLogVar
from utils import get_model, save_data, get_data
from model_prediction import agent_uncertain_predict, update_item_async
from gym_env import MovingItem, MovingAgent

from stable_baselines3 import DQN

from concurrent.futures import ThreadPoolExecutor

#from gym_env import MovingAvoidanceEnv


def draw_item_async(item, surface,color=None):
    """Asynchronous draw function"""

    item.draw(surface,color)
    return item

data = []

SAVE = False
RENDER = True

# Screen dimensions
WIDTH, HEIGHT = 200, 200
ITEM_RADIUS = 15
ITEM_COUNT = 5
MAX_STEPS = 1000
AGENT_SPEED = 3.0

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 128, 0)

if RENDER:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('2D Moving Items - Prediction Agent')
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)  # default font, size 30

#env = MovingAvoidanceEnv()

def render(items):

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pass

executor = ThreadPoolExecutor(max_workers=4)

model = get_model(
    "j_10_5", 
    custom_objects={'nll_gaussian': nll_gaussian, 'ClippedLogVar': ClippedLogVar},
    safe_mode=False
)


def main():
    items = [MovingItem(add_noise=True) for i in range(ITEM_COUNT)]

    agent = MovingAgent(add_noise=False)

    #model = DQN.load("dqn_avoidance_agent")

    
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
        update_futures = [executor.submit(update_item_async, item) for item in items]

        # Wait for all updates to complete
        for future in update_futures:
            future.result()

        if SAVE:
            for item in items:
                data.append({'episode':episode,'frame':frame,'item':item.id, 'x':item.x, 'y':item.y, 'vx':item.vx, 'vy':item.vy})
        

        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            screen.fill(WHITE)

            # Check collisions
            min_dist = float("inf")
            closest_item = None
            for obs in items:
                if not isinstance(obs, MovingAgent):
                    dist = math.hypot(obs.x - agent.x, obs.y - agent.y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_item = obs
                        pygame.draw.circle(screen, BLUE, (int(agent.x), int(agent.y)), 3)  # agent center
                        pygame.draw.circle(screen, BLUE, (int(obs.x), int(obs.y)), 3)  # obstacle center

                    if min_dist < ITEM_RADIUS * 2:
                        score = -10
                    else:
                        score = 1.0



            draw_futures = []
            for item in items:
                if not isinstance(item, MovingAgent):
                    color = ORANGE if item is closest_item else None
                    draw_futures.append(executor.submit(draw_item_async, item, screen, color))
            #draw_futures = [executor.submit(draw_item_async, item, screen) for item in items if not isinstance(item, MovingAgent)]

            # Wait for all draws to complete
            for future in draw_futures:
                future.result()

            # Run predictions
            futures = [executor.submit(agent_uncertain_predict, model, item) for item in items]

            # Wait for all predictions to complete
            uncertainty_predictions = []
            for i, future in enumerate(futures):
                try:
                    pred_x, pred_y, std_x, std_y  = future.result()
                    uncertainty_predictions.append((pred_x, pred_y, std_x, std_y))
                except Exception as e:
                    print(f"Prediction error: {e}")

            for item in items:
                if isinstance(item, MovingAgent):
                    item.draw(screen, [item for item in items if not isinstance(item, MovingAgent)], uncertainty_predictions)

            #obs = agent.get_observation(items,uncertainty_predictions)

            #action, _states = model.predict(obs)

            #dx, dy = env._action_to_velocity(action)





            score_surface = font.render(f"Score: {score}", True, (0, 0, 0))  # black text
            score_rect = score_surface.get_rect(topright=(WIDTH - 10, 10))
            screen.blit(score_surface, score_rect)

            pos_text = f"Agent: ({int(agent.x)}, {int(agent.y)})"
            pos_surface = font.render(pos_text, True, (0, 0, 0))
            # position it just below the score
            pos_rect = pos_surface.get_rect(topright=(WIDTH - 10, 40))  # 40px down from top
            screen.blit(pos_surface, pos_rect)

            dist_surface = font.render(f"Dist: {min_dist:.2f}", True, (0, 0, 0))  # two decimals
            dist_rect = dist_surface.get_rect(topright=(WIDTH - 10, 70))  # below the others
            screen.blit(dist_surface, dist_rect)


            # Draw all predictions at once
            for i, (item, (pred_x, pred_y, std_x, std_y)) in enumerate(zip(items, uncertainty_predictions)):
                prediction_color = GREEN if item.add_noise else RED
                # Draw anti-aliased prediction circles
                pygame.draw.circle(screen, prediction_color, (int(pred_x), int(pred_y)), 5, 0)
                pygame.draw.circle(screen, BLACK, (int(pred_x), int(pred_y)), 5, 1)
                # Draw anti-aliased lines
                pygame.draw.line(screen, BLACK, (int(item.x), int(item.y)), (int(pred_x), int(pred_y)), 2)

                # Calculate angle between current position and prediction
                dx = pred_x - item.x
                dy = pred_y - item.y
                angle = math.atan2(dy, dx)

                # Calculate variance-based angle spread (inverse relationship)
                # Higher variance = smaller angle spread
                total_variance = std_x + std_y
                max_angle_spread = math.pi / 2  # 90 degrees total (45 degrees each side)
                angle_spread = max_angle_spread / (1 + total_variance)  # Inverse relationship

                # Draw pie slice
                points = [(int(item.x), int(item.y))]  # Start at current position
                
                # Add arc points
                steps = 20
                for i in range(steps + 1):
                    current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                    radius = math.sqrt(dx**2 + dy**2)  # Distance to prediction point
                    x = item.x + radius * math.cos(current_angle)
                    y = item.y + radius * math.sin(current_angle)
                    points.append((int(x), int(y)))
                
                points.append((int(item.x), int(item.y)))  # Close the polygon
                
                # Draw filled polygon with semi-transparency
                surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (128, 128, 128, 64), points)  # Light gray, semi-transparent
                screen.blit(surface, (0, 0))
                
                # Draw outline
                pygame.draw.polygon(screen, (128, 128, 128), points, 1)  # Solid gray outline
        

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