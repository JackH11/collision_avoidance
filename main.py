import pygame
import random
import sys
import math
import gym
from gym import spaces
import numpy as np

# Screen dimensions
WIDTH, HEIGHT = 800, 600
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

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Moving Items - Prediction Agent')
clock = pygame.time.Clock()

HISTORY_LENGTH = 5


class MovingItem:
    def __init__(self, add_noise=False):
        self.x = random.randint(ITEM_RADIUS, WIDTH - ITEM_RADIUS)
        self.y = random.randint(ITEM_RADIUS, HEIGHT - ITEM_RADIUS)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.add_noise = add_noise
        self.noise_timer = 0

    def update(self):
        # Add noise to velocity if enabled
        if self.add_noise:
            self.noise_timer += 1
            if self.noise_timer % 30 == 0:  # Add noise every 30 frames
                noise_angle = random.uniform(0, 2 * math.pi)
                noise_magnitude = random.uniform(0.5, 1.5)
                self.vx += math.cos(noise_angle) * noise_magnitude
                self.vy += math.sin(noise_angle) * noise_magnitude
                # Limit maximum speed
                speed = math.sqrt(self.vx**2 + self.vy**2)
                if speed > 6:
                    self.vx = (self.vx / speed) * 6
                    self.vy = (self.vy / speed) * 6
        
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls
        if self.x < ITEM_RADIUS or self.x > WIDTH - ITEM_RADIUS:
            self.vx *= -1
        if self.y < ITEM_RADIUS or self.y > HEIGHT - ITEM_RADIUS:
            self.vy *= -1

    def draw(self, surface):
        color = RED if self.add_noise else BLUE
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), ITEM_RADIUS)

    def get_position(self):
        return self.x, self.y


class MovingAgent(MovingItem):
    def __init__(self):
        super().__init__()

    def get_state(self):
        return [self.x, self.y, self.vx, self.vy]

class MovingAvoidanceEnv(gym.Env):
    def __init__(self):
        super(MovingAvoidanceEnv, self).__init__()

        self.observation_space = spaces.Box(low=0, high=1, shape=(ITEM_COUNT * 4 + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)  # 8 directions

        self.agent = MovingItem(add_noise=False)
        self.obstacles = []
        self.steps = 0

    def reset(self):
        self.agent = MovingItem(add_noise=False)
        self.agent.x = WIDTH // 2
        self.agent.y = HEIGHT // 2
        self.agent.vx = 0
        self.agent.vy = 0

        self.obstacles = []
        for _ in range(ITEM_COUNT):
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
                return self._get_obs(), -100, True, {}

        done = self.steps >= MAX_STEPS
        return self._get_obs(), 1.0, done, {}

    def _get_obs(self):
        agent_state = [self.agent.x / WIDTH, self.agent.y / HEIGHT]
        obstacle_states = []
        for obs in self.obstacles:
            x, y, vx, vy = obs.get_state()
            obstacle_states.extend([
                x / WIDTH, y / HEIGHT,
                vx / 6.0, vy / 6.0  # normalize
            ])
        return np.array(agent_state + obstacle_states, dtype=np.float32)

    def _action_to_velocity(self, action):
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        angle_rad = math.radians(angles[action])
        return AGENT_SPEED * math.cos(angle_rad), AGENT_SPEED * math.sin(angle_rad)


        

# Placeholder agent that predicts the next position (simple linear prediction)
def agent_predict(item):
    # Predicts next position based on current velocity
    pred_x = item.x + item.vx * 10  # Predict 10 frames ahead
    pred_y = item.y + item.vy * 10
    return pred_x, pred_y

def main():
    items = [MovingItem(add_noise=i < ITEM_COUNT // 2) for i in range(ITEM_COUNT)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        for item in items:
            item.update()
            item.draw(screen)
            # Draw agent prediction
            pred_x, pred_y = agent_predict(item)
            prediction_color = GREEN if item.add_noise else RED
            pygame.draw.circle(screen, prediction_color, (int(pred_x), int(pred_y)), 5)
            pygame.draw.line(screen, BLACK, (int(item.x), int(item.y)), (int(pred_x), int(pred_y)), 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 