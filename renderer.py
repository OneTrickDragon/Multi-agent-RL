"""
Pygame live grid renderer with dirty-rect updates.

Agents rendered as colored grid cells, not sprites.
Supports headless mode via RENDER_HEADLESS env var.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from config import WorldConfig
    from agent import Agent

# Attempt pygame import — graceful fallback for headless
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ── Color Palette ──────────────────────────────────────────────────────────

# Terrain colors (R, G, B)
TERRAIN_COLORS = {
    0: (30, 100, 180),    # WATER — deep blue
    1: (120, 170, 80),    # LAND  — olive green
    2: (40, 100, 50),     # FOREST — dark green
    3: (140, 130, 115),   # MOUNTAIN — stone grey
}

# Agent class colors
AGENT_COLORS = {
    0: (255, 220, 50),    # FARMER  — golden yellow
    1: (220, 50, 50),     # WARRIOR — crimson
    2: (50, 180, 220),    # TRADER  — cyan
    3: (200, 120, 255),   # LEADER  — lavender
}

GRID_LINE_COLOR = (30, 30, 30)
BG_COLOR = (15, 15, 15)
TEXT_COLOR = (240, 240, 240)


class Renderer:
    """
    Pygame grid renderer.

    Only redraws cells that changed since last frame (dirty rects).
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.cell_size = config.cell_size
        self.width_px = config.width * self.cell_size
        self.height_px = config.height * self.cell_size
        self._hud_height = 60
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._prev_terrain_hash: Optional[int] = None
        self._prev_agent_positions: Dict[int, Tuple[int, int, int]] = {}
        self._initialized = False

    def init(self) -> bool:
        """Initialize Pygame display. Returns False on failure."""
        if not HAS_PYGAME:
            print("[Renderer] pygame not installed — running headless.")
            return False
        if os.environ.get("RENDER_HEADLESS", "0") == "1":
            return False

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.width_px, self.height_px + self._hud_height)
        )
        pygame.display.set_caption("Multi-Agent RL Civilization")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 14)
        self._initialized = True
        self._prev_terrain_hash = None
        self._prev_agent_positions = {}
        return True

    def render(
        self,
        terrain: np.ndarray,
        food: np.ndarray,
        agents: List[Agent],
        info: dict,
    ) -> bool:
        """
        Draw the current state. Returns False if the user closes the window.
        """
        if not self._initialized:
            return True

        # ── Handle events ─────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        cs = self.cell_size
        dirty_rects: List[pygame.Rect] = []

        # ── Terrain (draw once or on change) ──────────────────────────
        t_hash = hash(terrain.data.tobytes())
        if t_hash != self._prev_terrain_hash:
            for x in range(self.config.width):
                for y in range(self.config.height):
                    tt = int(terrain[x, y])
                    color = TERRAIN_COLORS.get(tt, (100, 100, 100))
                    rect = pygame.Rect(x * cs, y * cs, cs, cs)
                    pygame.draw.rect(self._screen, color, rect)
                    if cs >= 6:
                        pygame.draw.rect(self._screen, GRID_LINE_COLOR, rect, 1)
            self._prev_terrain_hash = t_hash
            dirty_rects.append(pygame.Rect(0, 0, self.width_px, self.height_px))

        # ── Clear old agent positions ─────────────────────────────────
        current_positions: Dict[int, Tuple[int, int, int]] = {}
        for a in agents:
            if a.alive:
                current_positions[a.id] = (a.x, a.y, int(a.cls))

        # Erase agents that moved or died
        for aid, (ox, oy, _) in self._prev_agent_positions.items():
            if aid not in current_positions or current_positions[aid][:2] != (ox, oy):
                tt = int(terrain[ox, oy])
                color = TERRAIN_COLORS.get(tt, (100, 100, 100))
                rect = pygame.Rect(ox * cs, oy * cs, cs, cs)
                pygame.draw.rect(self._screen, color, rect)
                if cs >= 6:
                    pygame.draw.rect(self._screen, GRID_LINE_COLOR, rect, 1)
                dirty_rects.append(rect)

        # ── Draw agents ───────────────────────────────────────────────
        for aid, (ax, ay, cls_int) in current_positions.items():
            color = AGENT_COLORS.get(cls_int, (255, 255, 255))
            rect = pygame.Rect(ax * cs, ay * cs, cs, cs)
            pygame.draw.rect(self._screen, color, rect)
            dirty_rects.append(rect)

        self._prev_agent_positions = current_positions

        # ── HUD ───────────────────────────────────────────────────────
        hud_rect = pygame.Rect(0, self.height_px, self.width_px, self._hud_height)
        pygame.draw.rect(self._screen, BG_COLOR, hud_rect)

        step = info.get("step", 0)
        alive = info.get("alive", 0)
        cc = info.get("class_counts", {})
        line1 = f"Step: {step:>5}  |  Alive: {alive:>4}  |  F:{cc.get('FARMER',0)} W:{cc.get('WARRIOR',0)} T:{cc.get('TRADER',0)} L:{cc.get('LEADER',0)}"
        line2 = f"Avg Food: {info.get('avg_food', 0):.1f}  |  Avg HP: {info.get('avg_health', 0):.1f}"
        surf1 = self._font.render(line1, True, TEXT_COLOR)
        surf2 = self._font.render(line2, True, TEXT_COLOR)
        self._screen.blit(surf1, (8, self.height_px + 6))
        self._screen.blit(surf2, (8, self.height_px + 26))
        dirty_rects.append(hud_rect)

        # ── Flip ──────────────────────────────────────────────────────
        pygame.display.update(dirty_rects)
        self._clock.tick(self.config.fps)
        return True

    def close(self) -> None:
        if self._initialized:
            pygame.quit()
            self._initialized = False
