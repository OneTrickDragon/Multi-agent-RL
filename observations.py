"""
Observation builder — constructs per-agent local views of the world.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agent import Agent
    from config import WorldConfig


def build_observation(
    agent: Agent,
    terrain: np.ndarray,
    food: np.ndarray,
    materials: np.ndarray,
    agents: list[Agent],
    config: WorldConfig,
    step: int,
    total_alive: int,
) -> dict:
    """
    Build a structured observation for a single agent.

    Returns a dict with:
        - "self": agent's own state vector
        - "local_terrain": (2R+1, 2R+1) terrain grid
        - "local_food": (2R+1, 2R+1) food levels
        - "local_materials": (2R+1, 2R+1) material levels
        - "local_agents": (2R+1, 2R+1, 3) — [class, health_frac, is_present]
        - "global": [total_alive, step]
    """
    r = config.observation_radius
    w, h = config.width, config.height
    window = 2 * r + 1

    # ── Self state ──────────────────────────────────────────────────────
    self_state = np.array([
        agent.cls,
        agent.health / config.starting_health,
        agent.food / max(config.starting_food, 1.0),
        agent.materials / max(config.max_inventory, 1.0),
        agent.age,
        agent.buff_multiplier,
    ], dtype=np.float32)

    # ── Local grid windows ──────────────────────────────────────────────
    # Pad arrays for edge-of-map agents
    pad_terrain = np.pad(terrain, r, mode="constant", constant_values=-1)
    pad_food = np.pad(food, r, mode="constant", constant_values=0)
    pad_mat = np.pad(materials, r, mode="constant", constant_values=0)

    # Offset by padding
    ax, ay = agent.x + r, agent.y + r
    local_terrain = pad_terrain[ax - r : ax + r + 1, ay - r : ay + r + 1].astype(np.float32)
    local_food = pad_food[ax - r : ax + r + 1, ay - r : ay + r + 1].astype(np.float32)
    local_materials = pad_mat[ax - r : ax + r + 1, ay - r : ay + r + 1].astype(np.float32)

    # ── Nearby agents channel ──────────────────────────────────────────
    local_agents = np.zeros((window, window, 3), dtype=np.float32)
    for other in agents:
        if not other.alive or other.id == agent.id:
            continue
        dx = other.x - agent.x + r
        dy = other.y - agent.y + r
        if 0 <= dx < window and 0 <= dy < window:
            local_agents[dx, dy, 0] = float(other.cls) + 1.0  # +1 so 0 = empty
            local_agents[dx, dy, 1] = other.health / config.starting_health
            local_agents[dx, dy, 2] = 1.0

    # ── Global signals ─────────────────────────────────────────────────
    global_info = np.array([total_alive, step], dtype=np.float32)

    return {
        "self": self_state,
        "local_terrain": local_terrain,
        "local_food": local_food,
        "local_materials": local_materials,
        "local_agents": local_agents,
        "global": global_info,
    }
