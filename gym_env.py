import pygame
from gym import Env
import random, uuid
import math
import numpy as np
from numpy.f2py.auxfuncs import throw_error
import asyncio
import math

from config import CONFIG
from utils import point_in_polygon
from gym import spaces
from model_prediction import make_simple_prediction

from concurrent.futures import ThreadPoolExecutor
from nn.nn import nll_gaussian, ClippedLogVar
from utils import get_model
from functools import partial

# Colors
WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
RED = tuple(CONFIG["colors"]["red"])
BLUE = tuple(CONFIG["colors"]["blue"])
GREEN = tuple(CONFIG["colors"]["green"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])
ORANGE = tuple(CONFIG["colors"]["orange"])
PURPLE = tuple(CONFIG["colors"]["purple"])


# Screen dimensions
WIDTH = CONFIG["boundary"]["width"]
HEIGHT = CONFIG["boundary"]["height"]
WINDOW_WIDTH = CONFIG["window"]["width"]
WINDOW_HEIGHT = CONFIG["window"]["height"]
ITEM_RADIUS = CONFIG["obstacle"]["radius"]
ITEM_COUNT = CONFIG["obstacle"]["count"]
AGENT_SPEED = CONFIG["agent"]["speed"]
MAX_STEPS = 1000
GOAL_RADIUS = CONFIG["goal"]["radius"]
GOAL_COLOR = tuple(CONFIG["goal"]["color"])

executor = ThreadPoolExecutor(max_workers=4)

model = get_model(
    "j_10_5",
    custom_objects={'nll_gaussian': nll_gaussian, 'ClippedLogVar': ClippedLogVar},
    safe_mode=False
)

def draw_item_async(item, surface,color=None):
    """Asynchronous draw function"""

    item.draw(surface,color)
    return item




class MovingAvoidanceEnv(Env):
    def __init__(self, render=False):
        super(MovingAvoidanceEnv, self).__init__()
        self.ITEM_COUNT = 5

        self.observation_space = spaces.Box(low=-1, high=4, shape=(3,30,30), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.agent = None
        self.obstacles = []
        self.steps = 0

        self.goal_x = None
        self.goal_y = None
        self.resetFood = True

        self.generate_goal()

        self.render = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 30)  # default font, size 30
            pygame.display.set_caption('2D Moving Items - Prediction Agent')


    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_goal(self):

        self.goal_x = random.randint(GOAL_RADIUS, WIDTH - GOAL_RADIUS)
        self.goal_y = random.randint(GOAL_RADIUS, HEIGHT - GOAL_RADIUS)

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

        self.goal_x = random.randint(GOAL_RADIUS, WIDTH - GOAL_RADIUS)
        self.goal_y = random.randint(GOAL_RADIUS, HEIGHT - GOAL_RADIUS)

        self.steps = 0
        return self._get_obs()

    def calculate_goal_distance(self, x, y):

        x_dist = abs(self.goal_x - x)
        y_dist = abs(self.goal_y - y)
        dist = math.hypot(x_dist, y_dist)

        return dist

    def calculate_goal_change_distance(self, agent):
        """
        Calculates the change in distance between the agent and the goal between this step and the last
        """
        if len(agent.xs) < 2:
            return 0

        d1 = self.calculate_goal_distance(agent.xs[-1], agent.ys[-1])
        d2 = self.calculate_goal_distance(agent.xs[-2], agent.ys[-2])
        distance_change = d1 - d2
        return distance_change


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

        done = self.steps >= MAX_STEPS

        dist = math.hypot(self.goal_x - self.agent.x, self.goal_y - self.agent.y)
        if dist < GOAL_RADIUS + ITEM_RADIUS:
            done = True
            return self._get_obs(), 100, True, {}

        for obs in self.obstacles:
            dist = math.hypot(obs.x - self.agent.x, obs.y - self.agent.y)
            if dist < ITEM_RADIUS * 2:
                done = True
                return self._get_obs(), -50, done, {}


        distance_change = self.calculate_goal_change_distance(self.agent)

        return self._get_obs(), distance_change - 5, done, {}


    def render(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



        self.screen.fill(WHITE)

        draw_func = partial(draw_item_async, surface=self.screen, color=LIGHT_GREY)
        list(executor.map(draw_func, self.obstacles))

        uncertainty_predictions = self.agent.make_predictions(self.obstacles)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen, obstacle)

        reward = self.get_reward(self.agent)
        text_surface = self.font.render(f"Reward: {reward}", True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.topright = (WIDTH - 10, 10)  # 10 px padding from the top-right corner
        self.screen.blit(text_surface, text_rect)

        self.draw_arrow_from_base(self.screen, BLACK, self.agent.x, self.agent.y, self.action)
        self.draw(self.screen, self, self.obstacles, uncertainty_predictions, True)

        self.draw_goal(self.screen)

        pygame.display.flip()
        self.clock.tick(120)


    def get_reward(self, agent):

        dist = math.hypot(self.goal_x - agent.x, self.goal_y - agent.y)
        if dist < GOAL_RADIUS + ITEM_RADIUS:
            return 100

        for obs in self.obstacles:
            dist = math.hypot(obs.x - agent.x, obs.y - agent.y)
            if dist < ITEM_RADIUS * 2:
                return -50

        distance_change = self.calculate_goal_change_distance(agent)

        return distance_change - 5


    def _get_obs(self):

        obs = self.agent.get_observation(self.obstacles, self)

        return obs

    def _action_to_angle(self, action):
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        return math.radians(angles[action])

    def _action_to_velocity(self, action):
        angle_rad = self._action_to_angle(action)
        return AGENT_SPEED * math.cos(angle_rad), AGENT_SPEED * math.sin(angle_rad)

    def draw_goal(self, surface):
        if not self.goal_x or not self.goal_y:
            self.generate_goal()
        pygame.draw.circle(surface, GOAL_COLOR, (WINDOW_WIDTH/2 - WIDTH/2 + self.goal_x, WINDOW_HEIGHT/2 - HEIGHT/2 + self.goal_y), GOAL_RADIUS)

    def draw_predictions(self, surface, items, predictions):

        # Draw all predictions at once
        for i, (item, (pred_x, pred_y, std_x, std_y)) in enumerate(zip(items, predictions)):
            prediction_color = GREEN if item.add_noise else RED
            # Draw anti-aliased prediction circles
            pygame.draw.circle(surface, prediction_color, (WINDOW_WIDTH/2 - WIDTH/2 + int(pred_x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(pred_y)), 5, 0)
            pygame.draw.circle(surface, BLACK, (WINDOW_WIDTH/2 - WIDTH/2 + int(pred_x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(pred_y)), 5, 1)
            # Draw anti-aliased lines
            pygame.draw.line(surface, BLACK, (WINDOW_WIDTH/2 - WIDTH/2 + int(item.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(item.y)), (WINDOW_WIDTH/2 - WIDTH/2 + int(pred_x), int(WINDOW_HEIGHT/2 - HEIGHT/2 + pred_y)), 2)

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
            points = [(WINDOW_WIDTH/2 - WIDTH/2 + int(item.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(item.y))]  # Start at current position

            # Add arc points
            steps = 20
            for i in range(steps + 1):
                current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                radius = math.sqrt(dx ** 2 + dy ** 2)  # Distance to prediction point
                x = item.x + radius * math.cos(current_angle)
                y = item.y + radius * math.sin(current_angle)
                points.append((WINDOW_WIDTH/2 - WIDTH/2 + int(x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(y)))

            points.append((WINDOW_WIDTH/2 - WIDTH/2 + int(item.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(item.y)))  # Close the polygon

            # Draw filled polygon with semi-transparency
            #surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(surface, (128, 128, 128, 64), points)  # Light gray, semi-transparent
            surface.blit(surface, (0, 0))

            # Draw outline
            pygame.draw.polygon(surface, (128, 128, 128), points, 1)  # Solid gray outline

    def draw_arrow_from_base(self, surface, color, base_x, base_y, action, length=20, arrowhead_length=6,
                             arrowhead_angle=30, width=2):
        # Convert angle to radians

        base_x = WINDOW_WIDTH/2 - WIDTH/2 + base_x
        base_y = WINDOW_HEIGHT/2 - HEIGHT/2 + base_y

        angle_rad = self._action_to_angle(action)

        # Calculate the end of the arrow shaft
        end_x = base_x + length * math.cos(angle_rad)
        end_y = base_y + length * math.sin(angle_rad)

        # Draw the shaft
        pygame.draw.line(surface, color, (base_x, base_y), (end_x,end_y), width)

        # Calculate the two arrowhead points
        left_angle = angle_rad + math.radians(180 - arrowhead_angle)
        right_angle = angle_rad - math.radians(180 - arrowhead_angle)

        left = (end_x + arrowhead_length * math.cos(left_angle),
                end_y + arrowhead_length * math.sin(left_angle))
        right = (end_x + arrowhead_length * math.cos(right_angle),
                 end_y + arrowhead_length * math.sin(right_angle))

        # Draw the arrowhead as a filled triangle
        pygame.draw.polygon(surface, color, [(end_x, end_y), left, right])

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
        pygame.draw.circle(surface, color, (WINDOW_WIDTH/2 - WIDTH/2 + int(self.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(self.y)), ITEM_RADIUS, 0)
        # Add a subtle outline for better definition
        pygame.draw.circle(surface, BLACK, (WINDOW_WIDTH/2 - WIDTH/2 + int(self.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(self.y)), ITEM_RADIUS, 1)

    def get_position(self):
        return self.x, self.y

    def get_history(self, lag=5,window=10):
        # Get the last 'lag' values for each feature
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

    def get_observation(self, items, env):

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

                        goal_dist = env.calculate_goal_distance(dot_x, dot_y)

                        if goal_dist < (dot_radius + GOAL_RADIUS):
                            cell_value = 4
                            cell_vx, cell_vy = 0,0
                            break

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

    def draw(self, surface,env, items=None,predictions=None, dots=False):
        color = LIGHT_GREY
        # Use anti-aliased circle for smoother appearance
        pygame.draw.circle(surface, color, (WINDOW_WIDTH/2 - WIDTH/2 + int(self.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(self.y)), ITEM_RADIUS, 0)
        # Add a subtle outline for better definition
        pygame.draw.circle(surface, BLACK, (WINDOW_WIDTH/2 - WIDTH/2 + int(self.x), WINDOW_HEIGHT/2 - HEIGHT/2 + int(self.y)), ITEM_RADIUS, 1)

        # --- Add red dots in a 10x10 square around the item ---
        dot_color = BLUE  # red
        dot_radius = 2  # radius of each dot
        spacing = 5  # distance between dots in pixels
        half_count = 15  # 10x10 square means 5 dots in each direction from center


        if not items or not predictions:
            return

        if dots:
            grid = self.get_observation(items, env)[0]

            for dx in range(-half_count, half_count):
                for dy in range(-half_count, half_count):
                    dot_x = WINDOW_WIDTH/2 - WIDTH/2 + int(self.x + dx * spacing)
                    dot_y = WINDOW_HEIGHT/2 - HEIGHT/2 + int(self.y + dy * spacing)

                    color_value = grid[dx + half_count, dy + half_count]
                    if color_value == 0:
                        dot_color = GREEN
                    elif color_value == 1:
                        dot_color = ORANGE
                    elif color_value == 2:
                        dot_color = RED
                    elif color_value == 3:
                        dot_color = BLACK
                    elif color_value == 4:
                        dot_color = PURPLE
                    else:
                        throw_error("Invalid color")


                    pygame.draw.circle(surface, dot_color, (dot_x, dot_y), dot_radius)

    async def make_prediction(self,items):

        tasks = [asyncio.to_thread(self.prediction_model,item) for item in items]
        return await asyncio.gather(*tasks)

    def make_predictions(self,items):

        return asyncio.run(self.make_prediction(items))





