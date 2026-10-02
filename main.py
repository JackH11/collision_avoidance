"""
Phase 5 demo — canonical interactive renderer for collision avoidance.

Gymnasium ``env.render()`` remains a stub; all visuals live here.

Modes:
  demo      Cleaner view + HUD counters (default). Grid dots off.
  research  Policy-grid dots on; same HUD (debug / teaching).

Examples:
  python main.py
  python main.py --mode demo --scenario baseline --seed 0
  python main.py --model models/qrdqn_CnnPolicy_easy_s0_cont/best_model.zip --algo qrdqn
  python main.py --headless --record media/demo.gif --episodes 3 --fps 30

Keys (windowed):
  G  toggle policy grid dots
  P  toggle prediction cones
  N  toggle obstacle velocity noise (applies next episode)
  R  reset episode now
  Space  pause
  + / -  zoom in / out
  Esc / quit  exit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Headless must set SDL driver before pygame import.
_HEADLESS_FLAG = "--headless" in sys.argv
if _HEADLESS_FLAG:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from config import CONFIG
from gym_env import MovingAvoidanceEnv
from model_loader import load_model
from model_prediction import resolve_prediction_backend

DEFAULT_MODEL_CANDIDATES = [
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


@dataclass
class HudStats:
    success: int = 0
    collision: int = 0
    timeout: int = 0
    episodes: int = 0
    ep_return: float = 0.0
    last_outcome: str = "—"

    def record(self, info: dict, ep_return: float) -> None:
        self.episodes += 1
        self.ep_return = ep_return
        if info.get("goal_reached"):
            self.success += 1
            self.last_outcome = "SUCCESS"
        elif info.get("collision"):
            self.collision += 1
            self.last_outcome = "COLLISION"
        elif info.get("timeout"):
            self.timeout += 1
            self.last_outcome = "TIMEOUT"
        else:
            self.last_outcome = "OTHER"


@dataclass
class ViewOpts:
    show_grid: bool = False
    show_predictions: bool = True
    obstacle_noise: Optional[bool] = None  # None = scenario default
    paused: bool = False
    zoom: float = 3.0  # display scale; crops to playfield then enlarges


@dataclass
class FrameRecorder:
    """Write RGB frames to GIF (Pillow) or MP4 (ffmpeg pipe)."""

    path: Path
    fps: int = 30
    frames: List[np.ndarray] = field(default_factory=list)
    _ffmpeg: Optional[subprocess.Popen] = None
    _tmp_dir: Optional[tempfile.TemporaryDirectory] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.path.suffix.lower()
        if suffix == ".mp4":
            self._ffmpeg = None
        elif suffix not in (".gif", ".mp4"):
            raise ValueError(f"Unsupported record format: {self.path} (use .gif or .mp4)")

    def _ensure_ffmpeg(self, w: int, h: int) -> None:
        if self._ffmpeg is not None:
            return
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def add(self, surface: pygame.Surface) -> None:
        # pygame surfarray is (w, h, 3); image/ffmpeg want (h, w, 3)
        arr = pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()
        if self.path.suffix.lower() == ".mp4":
            h, w = arr.shape[:2]
            self._ensure_ffmpeg(w, h)
            assert self._ffmpeg and self._ffmpeg.stdin
            self._ffmpeg.stdin.write(arr.tobytes())
        else:
            self.frames.append(arr)

    def close(self) -> None:
        if self.path.suffix.lower() == ".mp4":
            if self._ffmpeg is not None:
                if self._ffmpeg.stdin:
                    self._ffmpeg.stdin.close()
                self._ffmpeg.wait(timeout=120)
            return
        if not self.frames:
            return
        from PIL import Image

        images = [Image.fromarray(f) for f in self.frames]
        # Cap GIF size: subsample if very long
        max_frames = 240
        if len(images) > max_frames:
            step = max(1, len(images) // max_frames)
            images = images[::step]
        duration_ms = int(1000 / max(1, self.fps))
        images[0].save(
            self.path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )


def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(
        description="Collision-avoidance demo (Phase 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=None, help="SB3 zip (auto-picks Phase 2 best if omitted)")
    p.add_argument("--algo", default=None, choices=["ppo", "qrdqn", "dqn"])
    p.add_argument("--scenario", default="baseline", choices=["easy", "baseline", "hard"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--predictor",
        default="simple",
        choices=["simple", "nn", "nn_uncertainty"],
        help="Trajectory predictor backend",
    )
    p.add_argument(
        "--mode",
        default="demo",
        choices=["demo", "research"],
        help="demo = clean HUD; research = policy grid dots on",
    )
    p.add_argument("--reward-mode", default=None, choices=["old", "new"])
    p.add_argument("--episodes", type=int, default=0, help="Stop after N finished episodes (0 = unlimited)")
    p.add_argument("--max-frames", type=int, default=0, help="Hard frame cap (0 = unlimited)")
    p.add_argument("--fps", type=int, default=60, help="Display / record FPS")
    p.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="Visual scale: crop to the playfield then enlarge (physics/obs unchanged)",
    )
    p.add_argument("--record", default=None, help="Write .gif or .mp4 to this path")
    p.add_argument(
        "--headless",
        action="store_true",
        help="No window (SDL dummy). Implies recording if --record set; else writes media/demo.gif",
    )
    p.add_argument("--collect-data", action="store_true", help="Dump obstacle trajectories (research)")
    return p.parse_args(argv)


def _playfield_rect(env: MovingAvoidanceEnv, pad: int = 20) -> pygame.Rect:
    """Axis-aligned crop around the sim boundary (plus pad)."""
    ox = env.window_width / 2 - env.width / 2
    oy = env.window_height / 2 - env.height / 2
    return pygame.Rect(
        max(0, int(ox) - pad),
        max(0, int(oy) - pad),
        int(env.width) + 2 * pad,
        int(env.height) + 2 * pad,
    ).clip(pygame.Rect(0, 0, env.window_width, env.window_height))


def _draw_boundary(surface, env, color=(200, 200, 200)) -> None:
    ox = env.window_width / 2 - env.width / 2
    oy = env.window_height / 2 - env.height / 2
    pygame.draw.rect(
        surface,
        color,
        pygame.Rect(int(ox), int(oy), int(env.width), int(env.height)),
        2,
    )


def _draw_hud(
    surface,
    font,
    env,
    algo: str,
    stats: HudStats,
    view: ViewOpts,
    info: dict,
) -> None:
    black = (20, 20, 20)
    lines = [
        f"{algo.upper()}  |  {env.scenario_name}  |  {stats.last_outcome}",
        f"Ep {stats.episodes}  S:{stats.success}  C:{stats.collision}  T:{stats.timeout}",
        f"step {info.get('steps', 0)}  ret {stats.ep_return:.0f}  "
        f"goal_d {float(info.get('goal_dist', float('nan'))):.0f}  "
        f"clear {float(info.get('min_obstacle_dist', float('nan'))):.0f}",
        f"grid={'on' if view.show_grid else 'off'}  "
        f"cones={'on' if view.show_predictions else 'off'}  "
        f"noise={'on' if env.obstacle_noise else 'off'}  "
        f"zoom={view.zoom:.1f}x"
        + ("  PAUSED" if view.paused else ""),
    ]
    y = 8
    for text in lines:
        surf = font.render(text, True, black)
        # Light backing so HUD stays readable on busy frames
        pad = 2
        bg = pygame.Surface(
            (surf.get_width() + 2 * pad, surf.get_height() + 2 * pad),
            pygame.SRCALPHA,
        )
        bg.fill((255, 255, 255, 210))
        surface.blit(bg, (8 - pad, y - pad))
        surface.blit(surf, (8, y))
        y += surf.get_height() + 4


def _draw_world(
    surface,
    env: MovingAvoidanceEnv,
    action: int,
    predictions,
    view: ViewOpts,
) -> None:
    white = tuple(CONFIG["colors"]["white"])
    black = tuple(CONFIG["colors"]["black"])
    surface.fill(white)
    _draw_boundary(surface, env)

    if view.show_predictions and predictions is not None:
        env.draw_predictions(surface, env.obstacles, predictions)

    for obstacle in env.obstacles:
        obstacle.draw(surface)

    env.draw_arrow_from_base(surface, black, env.agent.x, env.agent.y, action)
    env.agent.draw(
        surface,
        env,
        env.obstacles,
        predictions,
        dots=view.show_grid,
    )
    env.draw_goal(surface)


def _compose_frame(
    world: pygame.Surface,
    env: MovingAvoidanceEnv,
    view: ViewOpts,
    font,
    algo: str,
    stats: HudStats,
    info: dict,
) -> pygame.Surface:
    """Crop to playfield, scale by zoom, then overlay HUD."""
    crop = _playfield_rect(env)
    clipped = world.subsurface(crop)
    zoom = max(1.0, float(view.zoom))
    out_size = (max(1, int(crop.width * zoom)), max(1, int(crop.height * zoom)))
    framed = pygame.transform.scale(clipped, out_size)
    _draw_hud(framed, font, env, algo, stats, view, info)
    return framed


def run_demo(args) -> int:
    model_path = args.model or _pick_default_model()
    headless = bool(args.headless)
    record_path = args.record
    if headless and not record_path:
        record_path = "media/demo.gif"

    # Episode budget for headless/record defaults so runs finish.
    episodes_limit = args.episodes
    if (headless or record_path) and episodes_limit <= 0:
        episodes_limit = 3
    max_frames = args.max_frames
    if (headless or record_path) and max_frames <= 0:
        max_frames = 90 * max(1, args.fps)  # ~90s safety cap

    view = ViewOpts(
        show_grid=(args.mode == "research"),
        show_predictions=True,
        zoom=max(1.0, float(args.zoom)),
    )

    reward_mode = args.reward_mode
    env = MovingAvoidanceEnv(
        scenario=args.scenario,
        reward_mode=reward_mode,
        prediction_backend=args.predictor,
    )
    if view.obstacle_noise is not None:
        env.obstacle_noise = view.obstacle_noise

    policy, algo_name, resolved = load_model(model_path, algo=args.algo)
    print(
        f"Demo mode={args.mode}  loaded {algo_name} from {resolved}  "
        f"scenario={env.scenario_name}  predictor={env.prediction_backend}  "
        f"seed={args.seed}  zoom={view.zoom:.1f}x"
    )

    pygame.init()
    pygame.display.set_caption("Collision Avoidance — Phase 5 Demo")
    world = pygame.Surface((env.window_width, env.window_height))
    crop0 = _playfield_rect(env)
    out_w = max(1, int(crop0.width * view.zoom))
    out_h = max(1, int(crop0.height * view.zoom))
    if headless:
        display = None
        screen = pygame.Surface((out_w, out_h))
    else:
        display = pygame.display.set_mode((out_w, out_h))
        screen = display
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("dejavusans", 20)

    recorder = FrameRecorder(Path(record_path), fps=min(args.fps, 30)) if record_path else None

    stats = HudStats()
    obs, info = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_idx = 0
    frame = 0
    action = 0
    predictions = None
    running = True
    collect_rows = []

    backend_name, predict_fn = resolve_prediction_backend(args.predictor)
    _ = backend_name
    # Keep agent prediction fn aligned with env backend for drawing cones.
    if hasattr(env.agent, "prediction_model"):
        env.agent.prediction_model = predict_fn

    try:
        while running:
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_g:
                            view.show_grid = not view.show_grid
                        elif event.key == pygame.K_p:
                            view.show_predictions = not view.show_predictions
                        elif event.key == pygame.K_n:
                            env.obstacle_noise = not env.obstacle_noise
                            view.obstacle_noise = env.obstacle_noise
                        elif event.key == pygame.K_SPACE:
                            view.paused = not view.paused
                        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                            view.zoom = min(6.0, view.zoom + 0.5)
                            crop = _playfield_rect(env)
                            size = (
                                max(1, int(crop.width * view.zoom)),
                                max(1, int(crop.height * view.zoom)),
                            )
                            display = pygame.display.set_mode(size)
                            screen = display
                        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                            view.zoom = max(1.0, view.zoom - 0.5)
                            crop = _playfield_rect(env)
                            size = (
                                max(1, int(crop.width * view.zoom)),
                                max(1, int(crop.height * view.zoom)),
                            )
                            display = pygame.display.set_mode(size)
                            screen = display
                        elif event.key == pygame.K_r:
                            stats.record(info, ep_return)
                            obs, info = env.reset(seed=args.seed + ep_idx + 1)
                            ep_idx += 1
                            ep_return = 0.0
                            if view.obstacle_noise is not None:
                                env.obstacle_noise = view.obstacle_noise

            if not view.paused:
                action_arr, _ = policy.predict(obs, deterministic=True)
                action = int(action_arr)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)

                if args.collect_data:
                    for obstacle in env.obstacles:
                        collect_rows.append(
                            {
                                "episode": ep_idx,
                                "frame": frame,
                                "item": getattr(obstacle, "id", None),
                                "x": obstacle.x,
                                "y": obstacle.y,
                                "vx": obstacle.vx,
                                "vy": obstacle.vy,
                            }
                        )

                if terminated or truncated:
                    stats.record(info, ep_return)
                    print(
                        f"episode {stats.episodes}: {stats.last_outcome}  "
                        f"return={ep_return:.1f}  steps={info.get('steps')}"
                    )
                    if episodes_limit and stats.episodes >= episodes_limit:
                        running = False
                        # Still draw the terminal frame below.
                    else:
                        obs, info = env.reset(seed=args.seed + stats.episodes)
                        ep_idx = stats.episodes
                        ep_return = 0.0
                        if view.obstacle_noise is not None:
                            env.obstacle_noise = view.obstacle_noise

            if view.show_predictions or view.show_grid:
                predictions = env.agent.make_predictions(env.obstacles)
            else:
                predictions = None

            _draw_world(world, env, action, predictions, view)
            frame_surf = _compose_frame(
                world, env, view, font, algo_name, stats, info
            )
            if display is not None:
                if frame_surf.get_size() != display.get_size():
                    display = pygame.display.set_mode(frame_surf.get_size())
                    screen = display
                display.blit(frame_surf, (0, 0))
                pygame.display.flip()
            else:
                screen = frame_surf
            if recorder is not None:
                recorder.add(frame_surf)

            clock.tick(args.fps)
            frame += 1
            if max_frames and frame >= max_frames:
                print(f"Reached --max-frames={max_frames}")
                running = False

    finally:
        if recorder is not None:
            recorder.close()
            print(f"Wrote recording: {recorder.path}")
        if args.collect_data and collect_rows:
            from utils import save_data

            save_data(
                collect_rows,
                fieldnames=["episode", "frame", "item", "x", "y", "vx", "vy"],
            )
            print(f"Wrote {len(collect_rows)} trajectory rows")
        env.close()
        pygame.quit()

    print(
        f"Done. episodes={stats.episodes} success={stats.success} "
        f"collision={stats.collision} timeout={stats.timeout}"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
