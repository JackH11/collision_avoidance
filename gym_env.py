import pygame
from gym import Env
import random, uuid
import math
import numpy as np
from utils import point_in_polygon

from stable_baselines3 import DQN

from gym import spaces

from model_prediction import agent_uncertain_predict
from concurrent.futures import ThreadPoolExecutor
from nn.nn import nll_gaussian, ClippedLogVar
from utils import get_model

import time
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
LIGHT_GREY = (211, 211, 211)

# Screen dimensions
WIDTH, HEIGHT = 200, 200
ITEM_RADIUS = 15
ITEM_COUNT = 5
MAX_STEPS = 1000
AGENT_SPEED = 3.0

executor = ThreadPoolExecutor(max_workers=4)

model = get_model(
    "j_10_5",
    custom_objects={'nll_gaussian': nll_gaussian, 'ClippedLogVar': ClippedLogVar},
    safe_mode=False
)


class MovingAvoidanceEnv(Env):
    def __init__(self):
        super(MovingAvoidanceEnv, self).__init__()
        self.ITEM_COUNT = 5

        self.observation_space = spaces.Box(low=0, high=2, shape=(30,30), dtype=np.uint8)
        self.action_space = spaces.Discrete(8)  # 8 directions

        self.agent = None
        self.obstacles = []
        self.steps = 0

    def seed(self, seed=None):
        # optional: use gym.utils.seeding to set up RNG
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent = MovingAgent()
        self.agent.x = WIDTH // 2
        self.agent.y = HEIGHT // 2
        self.agent.vx = 0
        self.agent.vy = 0

        self.obstacles = []
        for _ in range(self.ITEM_COUNT):
            item = MovingItem(add_noise=True)
            self.obstacles.append(item)

        self.steps = 0
        return self._get_obs()

    def step(self, action):
        self.steps += 1

        # Convert action to velocity
        dx, dy = self._action_to_velocity(action)
        self.agent.vx = dx
        self.agent.vy = dy

        # Update agent
        self.agent.update()

        # Update obstacles
        for obs in self.obstacles:
            obs.update()

        # Check collisions
        for obs in self.obstacles:
            dist = math.hypot(obs.x - self.agent.x, obs.y - self.agent.y)
            if dist < ITEM_RADIUS * 2:
                return self._get_obs(), -10, True, {}

        done = self.steps >= MAX_STEPS
        return self._get_obs(), 1.0, done, {}

    def _get_obs(self):

        futures = [executor.submit(agent_uncertain_predict, model, item) for item in self.obstacles]

        predictions = [f.result() for f in futures]

        obs = self.agent.get_observation(self.obstacles,predictions)

        return obs

    def _action_to_velocity(self, action):
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        angle_rad = math.radians(angles[action])
        return AGENT_SPEED * math.cos(angle_rad), AGENT_SPEED * math.sin(angle_rad)

class MovingItem:

    def __init__(self, add_noise=False):
        
        # kinematics
        self.id = str(uuid.uuid4())
        self.x = random.randint(ITEM_RADIUS, WIDTH - ITEM_RADIUS)
        self.y = random.randint(ITEM_RADIUS, HEIGHT - ITEM_RADIUS)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        # noise
        self.add_noise = add_noise
        self.noise_timer = 0
        self.noise_angle = random.uniform(0, 2*math.pi)
        self.noise_delta = random.uniform(-0.05, 0.05)

        # history
        self.xs: list[float] = [self.x]
        self.ys: list[float] = [self.y]
        self.vxs: list[float] = [self.vx]
        self.vys: list[float] = [self.vy]

    def update(self):
        # Add noise to velocity if enabled
        if self.add_noise:
            self.noise_timer += 1

            self.noise_angle += self.noise_delta

            if self.noise_timer % 120 == 0:
                self.noise_delta = random.uniform(-0.05, 0.05)

            base_noise_mag = 0.2
            self.vx += math.cos(self.noise_angle) * base_noise_mag
            self.vy += math.sin(self.noise_angle) * base_noise_mag


            # occasional larger jitter every so often
            if self.noise_timer % 90 == 0:
                jitter_angle = random.uniform(0, 2 * math.pi)
                jitter_mag = random.uniform(0.8, 2.0)
                self.vx += math.cos(jitter_angle) * jitter_mag
                self.vy += math.sin(jitter_angle) * jitter_mag

            # apply a slight damping so speed doesn't explode over time
            damping = 0.98
            self.vx *= damping
            self.vy *= damping

            # cap maximum speed
            speed = math.sqrt(self.vx**2 + self.vy**2)
            max_speed = 6.0
            if speed > max_speed:
                self.vx = (self.vx / speed) * max_speed
                self.vy = (self.vy / speed) * max_speed
        
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x < ITEM_RADIUS or self.x > WIDTH - ITEM_RADIUS:
            self.vx *= -1
        if self.y < ITEM_RADIUS or self.y > HEIGHT - ITEM_RADIUS:
            self.vy *= -1
        
        self.xs.append(self.x)
        self.ys.append(self.y)
        self.vxs.append(self.vx)
        self.vys.append(self.vy)

    def draw(self, surface, color=None):
        color = RED if self.add_noise else BLUE

        if color is not None:
            color = color
        # Use anti-aliased circle for smoother appearance
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), ITEM_RADIUS, 0)
        # Add a subtle outline for better definition
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), ITEM_RADIUS, 1)

    def get_position(self):
        return self.x, self.y

    def get_history(self, lag=5,window=10):
        # Get the last 'lag' values for each feature
        #if len(self.xs) < lag+window:
            #return np.zeros(lag*4)
        x_history = self.xs[-window:]
        y_history = self.ys[-window:]
        vx_history = self.vxs[-window:]
        vy_history = self.vys[-window:]
        
        # Pad with zeros if we don't have enough history
        while len(x_history) < window:
            x_history.insert(0, 0.0)
            y_history.insert(0, 0.0)
            vx_history.insert(0, 0.0)
            vy_history.insert(0, 0.0)
        
        # Flatten into the format: [x_lag_1, y_lag_1, vx_lag_1, vy_lag_1, x_lag_2, y_lag_2, ...]
        features = []
        for i in range(window-1,-1,-1):
            features.extend([x_history[i], y_history[i], vx_history[i], vy_history[i]])
        
        return features

class MovingAgent(MovingItem):
    def __init__(self, add_noise=False):
        super().__init__(add_noise=add_noise)

    def get_state(self):
        return [self.x, self.y, self.vx, self.vy]

    def get_observation(self, items, predictions):

        # --- Add red dots in a 10x10 square around the item ---
        dot_color = BLUE  # red
        dot_radius = 2  # radius of each dot
        spacing = 5  # distance between dots in pixels
        half_count = 15  # 10x10 square means 5 dots in each direction from center
        full_count = half_count * 2
        grid = np.zeros((full_count, full_count), dtype=np.uint8)

        for dx in range(-half_count, half_count):
            for dy in range(-half_count, half_count):
                # skip the center itself if you don't want to cover the item
                if dx == 0 and dy == 0:
                    continue
                dot_x = int(self.x + dx * spacing)
                dot_y = int(self.y + dy * spacing)
                dot_color = BLUE
                cell_value = 0
                for i,item in enumerate(items):  # a list of objects with .x and .y
                    dist_sq = (dot_x - item.x) ** 2 + (dot_y - item.y) ** 2
                    # Compare squared distances to avoid sqrt
                    if dist_sq < (dot_radius + ITEM_RADIUS) ** 2:
                        dot_color = (0, 255, 0)  # green if overlapping
                        cell_value = 2
                        break  # no need to check further

                    points = [(int(item.x), int(item.y))]  # Start at current position

                    # Calculate angle between current position and prediction
                    pred_x, pred_y, std_x, std_y = predictions[i]
                    dx1 = pred_x - item.x
                    dy1 = pred_y - item.y
                    angle = math.atan2(dy1, dx1)

                    # Calculate variance-based angle spread (inverse relationship)
                    # Higher variance = smaller angle spread
                    total_variance = std_x + std_y
                    max_angle_spread = math.pi / 2  # 90 degrees total (45 degrees each side)
                    angle_spread = max_angle_spread / (1 + total_variance)  # Inverse relationship

                    # Add arc points
                    steps = 20
                    for i in range(steps + 1):
                        current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                        radius = math.sqrt(dx1 ** 2 + dy1 ** 2)  # Distance to prediction point
                        x = item.x + radius * math.cos(current_angle)
                        y = item.y + radius * math.sin(current_angle)
                        points.append((int(x), int(y)))

                    points.append((int(item.x), int(item.y)))  # Close the polygon

                    if point_in_polygon(dot_x, dot_y, points):
                        dot_color = (255, 200, 0)
                        cell_value = 1
                        break

                grid[dx + half_count, dy + half_count] = cell_value
        return grid

    def draw(self, surface,items,predictions):
        color = LIGHT_GREY
        # Use anti-aliased circle for smoother appearance
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), ITEM_RADIUS, 0)
        # Add a subtle outline for better definition
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), ITEM_RADIUS, 1)

        # --- Add red dots in a 10x10 square around the item ---
        dot_color = BLUE  # red
        dot_radius = 2  # radius of each dot
        spacing = 5  # distance between dots in pixels
        half_count = 15  # 10x10 square means 5 dots in each direction from center

        for dx in range(-half_count, half_count + 1):
            for dy in range(-half_count, half_count + 1):
                # skip the center itself if you don't want to cover the item
                if dx == 0 and dy == 0:
                    continue
                dot_x = int(self.x + dx * spacing)
                dot_y = int(self.y + dy * spacing)
                dot_color = BLUE
                for i,item in enumerate(items):  # a list of objects with .x and .y
                    dist_sq = (dot_x - item.x) ** 2 + (dot_y - item.y) ** 2
                    # Compare squared distances to avoid sqrt
                    if dist_sq < (dot_radius + ITEM_RADIUS) ** 2:
                        dot_color = (0, 255, 0)  # green if overlapping
                        break  # no need to check further


                    points = [(int(item.x), int(item.y))]  # Start at current position

                    # Calculate angle between current position and prediction
                    pred_x, pred_y, std_x, std_y = predictions[i]
                    dx1 = pred_x - item.x
                    dy1 = pred_y - item.y
                    angle = math.atan2(dy1, dx1)

                    # Calculate variance-based angle spread (inverse relationship)
                    # Higher variance = smaller angle spread
                    total_variance = std_x + std_y
                    max_angle_spread = math.pi / 2  # 90 degrees total (45 degrees each side)
                    angle_spread = max_angle_spread / (1 + total_variance)  # Inverse relationship

                    # Add arc points
                    steps = 20
                    for i in range(steps + 1):
                        current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                        radius = math.sqrt(dx1 ** 2 + dy1 ** 2)  # Distance to prediction point
                        x = item.x + radius * math.cos(current_angle)
                        y = item.y + radius * math.sin(current_angle)
                        points.append((int(x), int(y)))

                    points.append((int(item.x), int(item.y)))  # Close the polygon

                    if point_in_polygon(dot_x, dot_y, points):
                        dot_color = (255, 200, 0)


                pygame.draw.circle(surface, dot_color, (dot_x, dot_y), dot_radius)
