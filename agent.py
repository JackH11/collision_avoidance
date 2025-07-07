import pygame
import math

class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 12
        self.color = (255, 255, 0)  # Yellow for ego
        self.speed = 2.5
        self.angle = 0  # In degrees
        self.radius = 12

    def update(self, items):
        perceived = self.perceive(items)
        predictions = self.predict(perceived)
        self.plan(predictions)
        self.move()

    def perceive(self, items):
        # Return all items within a certain radius
        return [item for item in items if self._distance_to(item) < 200]
    
    def predict(self, perceived_items):
        # Call agent_predict(item) for each
        return [(item, agent_predict(item)) for item in perceived_items]

    def plan(self, predicted_positions):
        # Stub: choose a direction away from predicted points
        pass

    def move(self):
        # Update position based on angle
        pass

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 0), (int(self.x), int(self.y)), self.radius)

    def _distance_to(self, item):
        dx = self.x - item.x
        dy = self.y - item.y
        return math.hypot(dx, dy)
