import pygame
from gym import Env
import random, uuid
import math
import numpy as np
from numpy.f2py.auxfuncs import throw_error
import asyncio

from config import CONFIG


from utils import point_in_polygon

from gym import spaces

from model_prediction import make_simple_prediction

from concurrent.futures import ThreadPoolExecutor
from nn.nn import nll_gaussian, ClippedLogVar
from utils import get_model

import time
# Colors
WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
RED = tuple(CONFIG["colors"]["red"])
BLUE = tuple(CONFIG["colors"]["blue"])
GREEN = tuple(CONFIG["colors"]["green"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])
ORANGE = tuple(CONFIG["colors"]["orange"])

# Screen dimensions
WIDTH, HEIGHT = 200, 200
ITEM_RADIUS = 8
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

        self.observation_space = spaces.Box(low=-1, high=3, shape=(3,30,30), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.agent = None
        self.obstacles = []
        self.steps = 0

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.agent = MovingAgent(make_simple_prediction)
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

        for obs in self.obstacles:
            dist = math.hypot(obs.x - self.agent.x, obs.y - self.agent.y)
            if dist < ITEM_RADIUS * 2:
                return self._get_obs(), -10, True, {}

        done = self.steps >= MAX_STEPS
        return self._get_obs(), 1.0, done, {}

    def _get_obs(self):

        obs = self.agent.get_observation(self.obstacles)

        return obs

    def _action_to_velocity(self, action):
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        angle_rad = math.radians(angles[action])
        return AGENT_SPEED * math.cos(angle_rad), AGENT_SPEED * math.sin(angle_rad)

    def draw_predictions(self, surface, items, predictions):

        # Draw all predictions at once
        for i, (item, (pred_x, pred_y, std_x, std_y)) in enumerate(zip(items, predictions)):
            prediction_color = GREEN if item.add_noise else RED
            # Draw anti-aliased prediction circles
            pygame.draw.circle(surface, prediction_color, (int(pred_x), int(pred_y)), 5, 0)
            pygame.draw.circle(surface, BLACK, (int(pred_x), int(pred_y)), 5, 1)
            # Draw anti-aliased lines
            pygame.draw.line(surface, BLACK, (int(item.x), int(item.y)), (int(pred_x), int(pred_y)), 2)

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
                radius = math.sqrt(dx ** 2 + dy ** 2)  # Distance to prediction point
                x = item.x + radius * math.cos(current_angle)
                y = item.y + radius * math.sin(current_angle)
                points.append((int(x), int(y)))

            points.append((int(item.x), int(item.y)))  # Close the polygon

            # Draw filled polygon with semi-transparency
            #surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(surface, (128, 128, 128, 64), points)  # Light gray, semi-transparent
            surface.blit(surface, (0, 0))

            # Draw outline
            pygame.draw.polygon(surface, (128, 128, 128), points, 1)  # Solid gray outline



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
        self.MAX_SPEED = 6

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
            if speed > self.MAX_SPEED:
                self.vx = (self.vx / speed) * self.MAX_SPEED
                self.vy = (self.vy / speed) * self.MAX_SPEED

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        # Bounce off walls
        if new_x < ITEM_RADIUS or new_x > WIDTH - ITEM_RADIUS:
            self.vx *= -1
        if new_y < ITEM_RADIUS or new_y > HEIGHT - ITEM_RADIUS:
            self.vy *= -1

        self.x += self.vx
        self.y += self.vy
        
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

    def get_normalize_velocity(self):

        return self.vx / self.MAX_SPEED, self.vy / self.MAX_SPEED

class MovingAgent(MovingItem):
    def __init__(self, prediction_model, add_noise=False):
        super().__init__(add_noise=add_noise)

        self.prediction_model = prediction_model


    def get_state(self):
        return [self.x, self.y, self.vx, self.vy]

    def get_observation(self, items):

        predictions = self.make_predictions(items)

        dot_radius = 2
        spacing = 5
        half_count = 15
        full_count = half_count * 2
        pos = np.zeros((full_count, full_count), dtype=np.float32)
        vx = np.zeros((full_count, full_count), dtype=np.float32)
        vy = np.zeros((full_count, full_count), dtype=np.float32)

        for dx in range(-half_count, half_count):
            for dy in range(-half_count, half_count):

                if dx == 0 and dy == 0:
                    continue
                dot_x = int(self.x + dx * spacing)
                dot_y = int(self.y + dy * spacing)

                cell_value = 0
                cell_vy = 0
                cell_vx = 0
                if dot_x < 0 or dot_x > WIDTH:
                    cell_value = 3
                    cell_vy = 0
                    cell_vx = 0
                if dot_y < 0 or dot_y > HEIGHT:
                    cell_value = 3
                    cell_vy = 0
                    cell_vx = 0

                if not cell_value == 3:
                    for i,item in enumerate(items):  # a list of objects with .x and .y
                        dist_sq = (dot_x - item.x) ** 2 + (dot_y - item.y) ** 2

                        if dist_sq < (dot_radius + ITEM_RADIUS) ** 2:
                            cell_value = 2
                            cell_vx, cell_vy = item.get_normalize_velocity()
                            break  # no need to check further

                        points = [(int(item.x), int(item.y))]  # Start at current position

                        pred_x, pred_y, std_x, std_y = predictions[i]
                        dx1 = pred_x - item.x
                        dy1 = pred_y - item.y
                        angle = math.atan2(dy1, dx1)

                        total_variance = std_x + std_y
                        max_angle_spread = math.pi / 2
                        angle_spread = max_angle_spread / (1 + total_variance)

                        steps = 20
                        for i in range(steps + 1):
                            current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                            radius = math.sqrt(dx1 ** 2 + dy1 ** 2)  # Distance to prediction point
                            x = item.x + radius * math.cos(current_angle)
                            y = item.y + radius * math.sin(current_angle)
                            points.append((int(x), int(y)))

                        points.append((int(item.x), int(item.y)))  # Close the polygon

                        if point_in_polygon(dot_x, dot_y, points):
                            cell_value = 1
                            cell_vx = 0
                            cell_vy = 0
                            break

                pos[dx + half_count, dy + half_count] = cell_value
                vx[dx + half_count, dy + half_count] = cell_vx
                vy[dx + half_count, dy + half_count] = cell_vy

        obs = np.stack([pos, vx, vy], axis=0)

        return obs

    def draw(self, surface,items=None,predictions=None):
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


        if not items or not predictions:
            return

        grid = self.get_observation(items)[0]

        for dx in range(-half_count, half_count):
            for dy in range(-half_count, half_count):
                dot_x = int(self.x + dx * spacing)
                dot_y = int(self.y + dy * spacing)

                color_value = grid[dx + half_count, dy + half_count]
                if color_value == 0:
                    dot_color = GREEN
                elif color_value == 1:
                    dot_color = ORANGE
                elif color_value == 2:
                    dot_color = RED
                elif color_value == 3:
                    dot_color = BLACK
                else:
                    throw_error("Invalid color")


                pygame.draw.circle(surface, dot_color, (dot_x, dot_y), dot_radius)

    async def make_prediction(self,items):

        tasks = [asyncio.to_thread(self.prediction_model,item) for item in items]
        return await asyncio.gather(*tasks)

    def make_predictions(self,items):

        return asyncio.run(self.make_prediction(items))





