"""
Gymnasium collision-avoidance environment.

Phase 1: vectorized local-grid observation, sync prediction backend,
reward A/B (old vs new), named scenarios, and rich step ``info``.

Interactive rendering lives in ``main.py``; ``env.render()`` is intentionally
a slim Gymnasium stub so the path is not half-broken.
"""

from __future__ import annotations

import math
import random
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium import Env, spaces

from config import CONFIG, resolve_scenario
from model_prediction import predict_batch, resolve_prediction_backend

# Colors
WHITE = tuple(CONFIG["colors"]["white"])
BLACK = tuple(CONFIG["colors"]["black"])
RED = tuple(CONFIG["colors"]["red"])
BLUE = tuple(CONFIG["colors"]["blue"])
GREEN = tuple(CONFIG["colors"]["green"])
LIGHT_GREY = tuple(CONFIG["colors"]["light_grey"])
ORANGE = tuple(CONFIG["colors"]["orange"])
PURPLE = tuple(CONFIG["colors"]["purple"])

# Screen / sim dimensions from active scenario config
WIDTH = CONFIG["boundary"]["width"]
HEIGHT = CONFIG["boundary"]["height"]
WINDOW_WIDTH = CONFIG["window"]["width"]
WINDOW_HEIGHT = CONFIG["window"]["height"]
ITEM_RADIUS = CONFIG["obstacle"]["radius"]
AGENT_SPEED = CONFIG["agent"]["speed"]
GOAL_RADIUS = CONFIG["goal"]["radius"]
GOAL_COLOR = tuple(CONFIG["goal"]["color"])

# Local grid obs layout (must stay (3, 30, 30) for baseline agents)
GRID_HALF = 15
GRID_SIZE = GRID_HALF * 2
GRID_SPACING = 5
GRID_DOT_RADIUS = 2

# Occupancy codes
CELL_EMPTY = 0
CELL_PRED = 1
CELL_OBSTACLE = 2
CELL_WALL = 3
CELL_GOAL = 4


def draw_item_async(item, surface, color=None):
    """Draw helper kept for main.py compatibility."""
    item.draw(surface, color)
    return item


class MovingAvoidanceEnv(Env):
    """
    Goal-seeking avoidance env with a local occupancy / prediction grid.

    Render: ``main.py`` is the canonical interactive renderer. Gymnasium
    ``render()`` is a no-op stub (see ``metadata``).
    """

    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(
        self,
        render_mode=None,
        prediction_backend=None,
        scenario=None,
        reward_mode=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.cfg = resolve_scenario(
            config if config is not None else CONFIG,
            scenario=scenario,
        )
        self.scenario_name = self.cfg.get("scenario", "baseline")

        obs_cfg = self.cfg["obstacle"]
        self.ITEM_COUNT = int(obs_cfg["count"])
        self.item_radius = float(obs_cfg["radius"])
        self.speed_min = float(obs_cfg.get("speed_min", 2.0))
        self.speed_max = float(obs_cfg.get("speed_max", 4.0))
        self.obstacle_max_speed = float(obs_cfg.get("max_speed", 6.0))
        self.obstacle_noise = bool(obs_cfg.get("noise", True))
        self.noise_scale = float(obs_cfg.get("noise_scale", 1.0))

        self.width = int(self.cfg["boundary"]["width"])
        self.height = int(self.cfg["boundary"]["height"])
        self.window_width = int(self.cfg["window"]["width"])
        self.window_height = int(self.cfg["window"]["height"])
        self.agent_speed = float(self.cfg["agent"]["speed"])
        self.goal_radius = float(self.cfg["goal"]["radius"])
        self.goal_color = tuple(self.cfg["goal"]["color"])
        self.max_steps = int(self.cfg.get("episode", {}).get("max_steps", 1000))

        reward_cfg = self.cfg.get("reward", {})
        self.reward_mode = (reward_mode or reward_cfg.get("mode", "old")).lower()
        if self.reward_mode not in ("old", "new"):
            raise ValueError("reward_mode must be 'old' or 'new'")
        self.progress_scale = float(reward_cfg.get("progress_scale", 1.0))
        self.step_cost = float(reward_cfg.get("step_cost", 0.1))
        self.near_miss_dist = float(reward_cfg.get("near_miss_dist", 24.0))
        self.near_miss_penalty = float(reward_cfg.get("near_miss_penalty", 0.5))

        backend_name, predict_fn = resolve_prediction_backend(
            prediction_backend
            or self.cfg.get("prediction", {}).get("backend", "simple")
        )
        self.prediction_backend = backend_name
        self._prediction_fn = predict_fn

        self.observation_space = spaces.Box(
            low=-1, high=4, shape=(3, GRID_SIZE, GRID_SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        self.agent = None
        self.obstacles: List[MovingItem] = []
        self.steps = 0
        self.goal_x = None
        self.goal_y = None
        self.resetFood = True
        self.last_action = 0
        self._render_warned = False

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        # Human display is owned by main.py; avoid opening a second window here.
        if render_mode == "human":
            warnings.warn(
                "MovingAvoidanceEnv.render_mode='human' is a stub. "
                "Use main.py for interactive Pygame rendering.",
                UserWarning,
                stacklevel=2,
            )

    def _goal_dist(self, x=None, y=None):
        if x is None:
            x = self.agent.x
        if y is None:
            y = self.agent.y
        return math.hypot(self.goal_x - x, self.goal_y - y)

    def _min_obstacle_dist(self):
        if not self.obstacles or self.agent is None:
            return float("inf")
        return min(
            math.hypot(obs.x - self.agent.x, obs.y - self.agent.y)
            for obs in self.obstacles
        )

    def _base_info(
        self,
        collision=False,
        goal_reached=False,
        timeout=False,
        action=None,
    ):
        goal_dist = (
            self._goal_dist()
            if self.agent is not None and self.goal_x is not None
            else float("nan")
        )
        min_obs = (
            self._min_obstacle_dist()
            if self.agent is not None
            else float("nan")
        )
        return {
            "collision": bool(collision),
            "goal_reached": bool(goal_reached),
            "timeout": bool(timeout),
            "goal_dist": float(goal_dist),
            "min_obstacle_dist": float(min_obs),
            "steps": int(self.steps),
            "action": int(self.last_action if action is None else action),
            "scenario": self.scenario_name,
            "reward_mode": self.reward_mode,
            "prediction_backend": self.prediction_backend,
        }

    def generate_goal(self):
        self.goal_x = random.randint(
            int(self.goal_radius), int(self.width - self.goal_radius)
        )
        self.goal_y = random.randint(
            int(self.goal_radius), int(self.height - self.goal_radius)
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent = MovingAgent(self._prediction_fn, env_ref=self)
        self.agent.x = self.width // 2
        self.agent.y = self.height // 2
        self.agent.vx = 0
        self.agent.vy = 0
        self.agent.xs = [self.agent.x]
        self.agent.ys = [self.agent.y]
        self.agent.vxs = [self.agent.vx]
        self.agent.vys = [self.agent.vy]
        self.agent.MAX_SPEED = self.obstacle_max_speed

        self.obstacles = []
        for _ in range(self.ITEM_COUNT):
            item = MovingItem(
                add_noise=self.obstacle_noise,
                speed_min=self.speed_min,
                speed_max=self.speed_max,
                max_speed=self.obstacle_max_speed,
                noise_scale=self.noise_scale,
                width=self.width,
                height=self.height,
                radius=self.item_radius,
            )
            self.obstacles.append(item)

        self.generate_goal()
        self.steps = 0
        self.last_action = 0

        obs = self._get_obs()
        return obs, self._base_info()

    def calculate_goal_distance(self, x, y):
        return math.hypot(self.goal_x - x, self.goal_y - y)

    def calculate_goal_change_distance(self, agent):
        """Signed change in goal distance: positive = moved closer (old reward)."""
        if len(agent.xs) < 2:
            return 0.0
        d1 = self.calculate_goal_distance(agent.xs[-1], agent.ys[-1])
        d2 = self.calculate_goal_distance(agent.xs[-2], agent.ys[-2])
        # Historical convention: d_new - d_old is inverted in old reward formula
        # which used distance_change - 5 where distance_change = d1 - d2
        # (negative when approaching). Keep that for mode=old.
        return d1 - d2

    def _compute_reward(self, goal_reached, collision, min_obs_dist):
        if goal_reached:
            return 100.0
        if collision:
            return -50.0

        distance_change = self.calculate_goal_change_distance(self.agent)

        if self.reward_mode == "old":
            # Unchanged vs baseline: Δgoal_distance − 5
            return float(distance_change - 5)

        # New: reward progress toward goal (negative Δd), small step cost,
        # optional near-miss penalty when clearance is tight.
        progress = -distance_change * self.progress_scale
        reward = progress - self.step_cost
        if min_obs_dist < self.near_miss_dist:
            # Stronger penalty the closer the miss.
            proximity = 1.0 - (min_obs_dist / self.near_miss_dist)
            reward -= self.near_miss_penalty * proximity
        return float(reward)

    def step(self, action):
        self.steps += 1
        self.last_action = int(action)

        dx, dy = self._action_to_velocity(action)
        self.agent.vx = dx
        self.agent.vy = dy
        self.agent.update()

        for obs in self.obstacles:
            obs.update()

        timeout = self.steps >= self.max_steps
        goal_dist = self._goal_dist()
        min_obs_dist = self._min_obstacle_dist()

        goal_reached = goal_dist < self.goal_radius + self.item_radius
        collision = min_obs_dist < self.item_radius * 2

        if goal_reached:
            reward = self._compute_reward(True, False, min_obs_dist)
            info = self._base_info(goal_reached=True, action=action)
            return self._get_obs(), reward, True, False, info

        if collision:
            reward = self._compute_reward(False, True, min_obs_dist)
            info = self._base_info(collision=True, action=action)
            return self._get_obs(), reward, True, False, info

        reward = self._compute_reward(False, False, min_obs_dist)
        info = self._base_info(timeout=timeout, action=action)
        return self._get_obs(), reward, False, timeout, info

    def render(self):
        """
        Gymnasium render stub.

        Interactive drawing is intentionally owned by ``main.py`` (same grid
        the policy sees). Calling this does not open a window or draw.
        """
        if self.render_mode == "human" and not self._render_warned:
            warnings.warn(
                "env.render() is a no-op; use main.py for the Pygame demo.",
                UserWarning,
                stacklevel=2,
            )
            self._render_warned = True
        return None

    def get_reward(self, agent):
        """Reward for an external agent (main.py demo path)."""
        dist = math.hypot(self.goal_x - agent.x, self.goal_y - agent.y)
        if dist < self.goal_radius + self.item_radius:
            return 100.0

        min_obs = float("inf")
        for obs in self.obstacles:
            d = math.hypot(obs.x - agent.x, obs.y - agent.y)
            min_obs = min(min_obs, d)
            if d < self.item_radius * 2:
                return -50.0

        # Temporarily bind histories for Δd if agent is not self.agent
        prev_agent = self.agent
        self.agent = agent
        try:
            reward = self._compute_reward(False, False, min_obs)
        finally:
            self.agent = prev_agent
        return reward

    def _get_obs(self):
        return self.agent.get_observation(self.obstacles, self)

    def _action_to_angle(self, action):
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        return math.radians(angles[int(action)])

    def _action_to_velocity(self, action):
        angle_rad = self._action_to_angle(action)
        return (
            self.agent_speed * math.cos(angle_rad),
            self.agent_speed * math.sin(angle_rad),
        )

    def draw_goal(self, surface):
        if not self.goal_x or not self.goal_y:
            self.generate_goal()
        pygame.draw.circle(
            surface,
            self.goal_color,
            (
                self.window_width / 2 - self.width / 2 + self.goal_x,
                self.window_height / 2 - self.height / 2 + self.goal_y,
            ),
            self.goal_radius,
        )

    def draw_predictions(self, surface, items, predictions):
        for item, (pred_x, pred_y, std_x, std_y) in zip(items, predictions):
            prediction_color = GREEN if item.add_noise else RED
            ox = self.window_width / 2 - self.width / 2
            oy = self.window_height / 2 - self.height / 2
            pygame.draw.circle(
                surface, prediction_color, (ox + int(pred_x), oy + int(pred_y)), 5, 0
            )
            pygame.draw.circle(
                surface, BLACK, (ox + int(pred_x), oy + int(pred_y)), 5, 1
            )
            pygame.draw.line(
                surface,
                BLACK,
                (ox + int(item.x), oy + int(item.y)),
                (ox + int(pred_x), oy + int(pred_y)),
                2,
            )

            dx = pred_x - item.x
            dy = pred_y - item.y
            angle = math.atan2(dy, dx)
            total_variance = std_x + std_y
            angle_spread = (math.pi / 2) / (1 + total_variance)
            radius = math.sqrt(dx ** 2 + dy ** 2)

            points = [(ox + int(item.x), oy + int(item.y))]
            steps = 20
            for i in range(steps + 1):
                current_angle = angle - angle_spread + (2 * angle_spread * i / steps)
                x = item.x + radius * math.cos(current_angle)
                y = item.y + radius * math.sin(current_angle)
                points.append((ox + int(x), oy + int(y)))
            points.append((ox + int(item.x), oy + int(item.y)))
            pygame.draw.polygon(surface, (128, 128, 128), points, 1)

    def draw_arrow_from_base(
        self,
        surface,
        color,
        base_x,
        base_y,
        action,
        length=20,
        arrowhead_length=6,
        arrowhead_angle=30,
        width=2,
    ):
        base_x = self.window_width / 2 - self.width / 2 + base_x
        base_y = self.window_height / 2 - self.height / 2 + base_y
        angle_rad = self._action_to_angle(action)
        end_x = base_x + length * math.cos(angle_rad)
        end_y = base_y + length * math.sin(angle_rad)
        pygame.draw.line(surface, color, (base_x, base_y), (end_x, end_y), width)

        left_angle = angle_rad + math.radians(180 - arrowhead_angle)
        right_angle = angle_rad - math.radians(180 - arrowhead_angle)
        left = (
            end_x + arrowhead_length * math.cos(left_angle),
            end_y + arrowhead_length * math.sin(left_angle),
        )
        right = (
            end_x + arrowhead_length * math.cos(right_angle),
            end_y + arrowhead_length * math.sin(right_angle),
        )
        pygame.draw.polygon(surface, color, [(end_x, end_y), left, right])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None


class MovingItem:
    def __init__(
        self,
        add_noise=False,
        speed_min=2.0,
        speed_max=4.0,
        max_speed=6.0,
        noise_scale=1.0,
        width=None,
        height=None,
        radius=None,
    ):
        self.id = str(uuid.uuid4())
        self.width = int(width if width is not None else WIDTH)
        self.height = int(height if height is not None else HEIGHT)
        self.radius = float(radius if radius is not None else ITEM_RADIUS)

        self.x = random.randint(int(self.radius), int(self.width - self.radius))
        self.y = random.randint(int(self.radius), int(self.height - self.radius))
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(speed_min, speed_max)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.MAX_SPEED = max_speed

        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.noise_timer = 0
        self.noise_angle = random.uniform(0, 2 * math.pi)
        self.noise_delta = random.uniform(-0.05, 0.05) * noise_scale

        self.xs: list[float] = [self.x]
        self.ys: list[float] = [self.y]
        self.vxs: list[float] = [self.vx]
        self.vys: list[float] = [self.vy]

    def update(self):
        if self.add_noise:
            self.noise_timer += 1
            self.noise_angle += self.noise_delta

            if self.noise_timer % 120 == 0:
                self.noise_delta = random.uniform(-0.05, 0.05) * self.noise_scale

            base_noise_mag = 0.2 * self.noise_scale
            self.vx += math.cos(self.noise_angle) * base_noise_mag
            self.vy += math.sin(self.noise_angle) * base_noise_mag

            if self.noise_timer % 90 == 0:
                jitter_angle = random.uniform(0, 2 * math.pi)
                jitter_mag = random.uniform(0.8, 2.0) * self.noise_scale
                self.vx += math.cos(jitter_angle) * jitter_mag
                self.vy += math.sin(jitter_angle) * jitter_mag

            self.vx *= 0.98
            self.vy *= 0.98

            speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if speed > self.MAX_SPEED:
                self.vx = (self.vx / speed) * self.MAX_SPEED
                self.vy = (self.vy / speed) * self.MAX_SPEED

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        if new_x < self.radius or new_x > self.width - self.radius:
            self.vx *= -1
        if new_y < self.radius or new_y > self.height - self.radius:
            self.vy *= -1

        self.x += self.vx
        self.y += self.vy

        self.xs.append(self.x)
        self.ys.append(self.y)
        self.vxs.append(self.vx)
        self.vys.append(self.vy)

    def draw(self, surface, color=None):
        draw_color = RED if self.add_noise else BLUE
        if color is not None and not isinstance(color, MovingItem):
            draw_color = color
        ox = WINDOW_WIDTH / 2 - WIDTH / 2
        oy = WINDOW_HEIGHT / 2 - HEIGHT / 2
        pygame.draw.circle(
            surface, draw_color, (ox + int(self.x), oy + int(self.y)), int(self.radius), 0
        )
        pygame.draw.circle(
            surface, BLACK, (ox + int(self.x), oy + int(self.y)), int(self.radius), 1
        )

    def get_position(self):
        return self.x, self.y

    def get_history(self, lag=5, window=10):
        x_history = self.xs[-window:]
        y_history = self.ys[-window:]
        vx_history = self.vxs[-window:]
        vy_history = self.vys[-window:]

        while len(x_history) < window:
            x_history.insert(0, 0.0)
            y_history.insert(0, 0.0)
            vx_history.insert(0, 0.0)
            vy_history.insert(0, 0.0)

        features = []
        for i in range(window - 1, -1, -1):
            features.extend(
                [x_history[i], y_history[i], vx_history[i], vy_history[i]]
            )
        return features

    def get_normalize_velocity(self):
        return self.vx / self.MAX_SPEED, self.vy / self.MAX_SPEED


class MovingAgent(MovingItem):
    def __init__(self, prediction_model, add_noise=False, env_ref=None):
        # Agent uses env dimensions when available.
        width = env_ref.width if env_ref is not None else WIDTH
        height = env_ref.height if env_ref is not None else HEIGHT
        radius = env_ref.item_radius if env_ref is not None else ITEM_RADIUS
        super().__init__(
            add_noise=add_noise, width=width, height=height, radius=radius
        )
        self.prediction_model = prediction_model
        self._env_ref = env_ref

    def get_state(self):
        return [self.x, self.y, self.vx, self.vy]

    def make_predictions(self, items):
        """Sync batch predictions (no asyncio.run)."""
        return predict_batch(items, predict_fn=self.prediction_model)

    def get_observation(self, items, env):
        """
        Vectorized (3, 30, 30) local grid.

        Channel 0 occupancy codes: 0 empty, 1 prediction cone, 2 obstacle,
        3 wall, 4 goal. Channels 1/2 carry normalized obstacle vx/vy on
        obstacle cells.
        """
        predictions = self.make_predictions(items)
        return build_local_grid_obs(
            agent_x=self.x,
            agent_y=self.y,
            items=items,
            predictions=predictions,
            goal_x=env.goal_x,
            goal_y=env.goal_y,
            width=getattr(env, "width", WIDTH),
            height=getattr(env, "height", HEIGHT),
            item_radius=getattr(env, "item_radius", ITEM_RADIUS),
            goal_radius=getattr(env, "goal_radius", GOAL_RADIUS),
        )

    def draw(self, surface, env, items=None, predictions=None, dots=False):
        color = LIGHT_GREY
        ox = WINDOW_WIDTH / 2 - WIDTH / 2
        oy = WINDOW_HEIGHT / 2 - HEIGHT / 2
        pygame.draw.circle(
            surface, color, (ox + int(self.x), oy + int(self.y)), int(ITEM_RADIUS), 0
        )
        pygame.draw.circle(
            surface, BLACK, (ox + int(self.x), oy + int(self.y)), int(ITEM_RADIUS), 1
        )

        if not items or not predictions:
            return

        if dots:
            grid = self.get_observation(items, env)[0]
            for dx in range(-GRID_HALF, GRID_HALF):
                for dy in range(-GRID_HALF, GRID_HALF):
                    dot_x = ox + int(self.x + dx * GRID_SPACING)
                    dot_y = oy + int(self.y + dy * GRID_SPACING)
                    color_value = grid[dx + GRID_HALF, dy + GRID_HALF]
                    if color_value == CELL_EMPTY:
                        dot_color = GREEN
                    elif color_value == CELL_PRED:
                        dot_color = ORANGE
                    elif color_value == CELL_OBSTACLE:
                        dot_color = RED
                    elif color_value == CELL_WALL:
                        dot_color = BLACK
                    elif color_value == CELL_GOAL:
                        dot_color = PURPLE
                    else:
                        raise ValueError(f"Invalid occupancy code: {color_value}")
                    pygame.draw.circle(surface, dot_color, (dot_x, dot_y), GRID_DOT_RADIUS)


def build_local_grid_obs(
    agent_x: float,
    agent_y: float,
    items,
    predictions,
    goal_x: float,
    goal_y: float,
    width: float = WIDTH,
    height: float = HEIGHT,
    item_radius: float = ITEM_RADIUS,
    goal_radius: float = GOAL_RADIUS,
) -> np.ndarray:
    """
    Rasterize the local occupancy / velocity grid with NumPy masks.

    Priority (high wins): wall > obstacle body > goal > prediction cone > empty.
    """
    offsets = (np.arange(GRID_SIZE) - GRID_HALF) * GRID_SPACING
    # indexing="ij" matches legacy pos[dx + half, dy + half] layout.
    gx, gy = np.meshgrid(
        agent_x + offsets,
        agent_y + offsets,
        indexing="ij",
    )

    pos = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    vx = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    vy = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # Center cell left empty (matches legacy skip of dx==0, dy==0).
    center = GRID_HALF

    # --- Prediction cones (lowest non-empty priority) ---
    for item, (pred_x, pred_y, std_x, std_y) in zip(items, predictions):
        dx = pred_x - item.x
        dy = pred_y - item.y
        radius = math.hypot(dx, dy)
        if radius < 1e-6:
            continue
        pred_angle = math.atan2(dy, dx)
        angle_spread = (math.pi / 2) / (1.0 + float(std_x) + float(std_y))

        rel_x = gx - item.x
        rel_y = gy - item.y
        dist = np.hypot(rel_x, rel_y)
        ang = np.arctan2(rel_y, rel_x)
        # Smallest absolute angle difference to prediction heading.
        ang_diff = np.abs((ang - pred_angle + math.pi) % (2 * math.pi) - math.pi)
        in_cone = (dist <= radius) & (dist > 0) & (ang_diff <= angle_spread)
        pos = np.where(in_cone, CELL_PRED, pos)

    # --- Goal stamp ---
    goal_dist = np.hypot(gx - goal_x, gy - goal_y)
    in_goal = goal_dist < (GRID_DOT_RADIUS + goal_radius)
    pos = np.where(in_goal, CELL_GOAL, pos)

    # --- Obstacle bodies + velocity channels ---
    hit_r2 = (GRID_DOT_RADIUS + item_radius) ** 2
    for item in items:
        dist_sq = (gx - item.x) ** 2 + (gy - item.y) ** 2
        in_body = dist_sq < hit_r2
        if not np.any(in_body):
            continue
        nvx, nvy = item.get_normalize_velocity()
        pos = np.where(in_body, CELL_OBSTACLE, pos)
        vx = np.where(in_body, nvx, vx)
        vy = np.where(in_body, nvy, vy)

    # --- Walls (highest priority) ---
    outside = (gx < 0) | (gx > width) | (gy < 0) | (gy > height)
    pos = np.where(outside, CELL_WALL, pos)
    vx = np.where(outside, 0.0, vx)
    vy = np.where(outside, 0.0, vy)

    # Legacy: leave the agent-centered cell empty / zero velocity.
    pos[center, center] = CELL_EMPTY
    vx[center, center] = 0.0
    vy[center, center] = 0.0

    return np.stack([pos, vx, vy], axis=0).astype(np.float32)
