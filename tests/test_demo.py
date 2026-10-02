"""CLI / recorder smoke tests for Phase 5 demo (no model zip required)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from main import FrameRecorder, ViewOpts, parse_args


def test_parse_args_defaults():
    args = parse_args([])
    assert args.mode == "demo"
    assert args.scenario == "baseline"
    assert args.predictor == "simple"
    assert args.seed == 0
    assert args.headless is False


def test_parse_args_research_and_record():
    args = parse_args(
        [
            "--mode",
            "research",
            "--headless",
            "--record",
            "media/out.gif",
            "--episodes",
            "2",
            "--fps",
            "15",
        ]
    )
    assert args.mode == "research"
    assert args.headless is True
    assert args.record == "media/out.gif"
    assert args.episodes == 2
    assert args.fps == 15


def test_view_opts_defaults():
    v = ViewOpts()
    assert v.show_grid is False
    assert v.show_predictions is True
    assert v.paused is False


def test_frame_recorder_gif(tmp_path: Path):
    pygame.init()
    surface = pygame.Surface((64, 48))
    surface.fill((255, 255, 255))
    pygame.draw.circle(surface, (0, 128, 255), (32, 24), 10)
    out = tmp_path / "tiny.gif"
    rec = FrameRecorder(out, fps=10)
    for _ in range(5):
        rec.add(surface)
    rec.close()
    pygame.quit()
    assert out.exists()
    assert out.stat().st_size > 50
