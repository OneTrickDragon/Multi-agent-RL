"""
Procedural terrain generation using Perlin noise.

Terrain types:
    0 — WATER
    1 — LAND
    2 — FOREST
    3 — MOUNTAIN
"""
from __future__ import annotations

import numpy as np
from enum import IntEnum


class TerrainType(IntEnum):
    WATER = 0
    LAND = 1
    FOREST = 2
    MOUNTAIN = 3


# ── Movement cost per terrain type ─────────────────────────────────────────
# 0.0 = impassable, 1.0 = normal, >1.0 = costly
TERRAIN_MOVE_COST = {
    TerrainType.WATER: 0.0,       # impassable
    TerrainType.LAND: 1.0,
    TerrainType.FOREST: 1.3,      # slightly slower
    TerrainType.MOUNTAIN: 2.0,    # very slow
}

# ── Resource yield multiplier per terrain ───────────────────────────────────
TERRAIN_FOOD_YIELD = {
    TerrainType.WATER: 0.0,
    TerrainType.LAND: 1.0,
    TerrainType.FOREST: 1.5,      # foraging bonus
    TerrainType.MOUNTAIN: 0.3,
}

TERRAIN_MATERIAL_YIELD = {
    TerrainType.WATER: 0.0,
    TerrainType.LAND: 0.8,
    TerrainType.FOREST: 0.6,
    TerrainType.MOUNTAIN: 2.0,    # mining bonus
}

# ── Combat modifier per terrain ─────────────────────────────────────────────
TERRAIN_DEFENSE_BONUS = {
    TerrainType.WATER: 0.0,
    TerrainType.LAND: 0.0,
    TerrainType.FOREST: 0.15,     # cover bonus
    TerrainType.MOUNTAIN: 0.25,   # high ground
}


# ── Perlin Noise Implementation ────────────────────────────────────────────
# Compact Perlin noise — no external dependency required.

def _fade(t: np.ndarray) -> np.ndarray:
    """Quintic smoothstep: 6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _perlin_2d(
    width: int,
    height: int,
    scale: float,
    octaves: int,
    seed: int,
) -> np.ndarray:
    """
    Generate a 2D Perlin noise array in [0, 1].

    Parameters
    ----------
    width, height : grid dimensions
    scale : base frequency (lower = larger blobs)
    octaves : number of noise layers
    seed : reproducibility seed

    Returns
    -------
    np.ndarray of shape (width, height) with values in [0, 1]
    """
    rng = np.random.RandomState(seed)
    noise = np.zeros((width, height), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0

    for _ in range(octaves):
        # Generate random gradient grid
        freq_w = max(2, int(width * scale))
        freq_h = max(2, int(height * scale))
        angles = rng.uniform(0, 2 * np.pi, (freq_w + 1, freq_h + 1))
        gradients_x = np.cos(angles)
        gradients_y = np.sin(angles)

        # Coordinate grids mapped to gradient space
        xs = np.linspace(0, freq_w - 1, width, endpoint=False)
        ys = np.linspace(0, freq_h - 1, height, endpoint=False)
        xg, yg = np.meshgrid(xs, ys, indexing="ij")

        x0 = xg.astype(int)
        y0 = yg.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # Fractional parts
        fx = xg - x0
        fy = yg - y0

        # Dot products with corner gradients
        def dot_grad(gx_idx, gy_idx, dx, dy):
            return gradients_x[gx_idx, gy_idx] * dx + gradients_y[gx_idx, gy_idx] * dy

        n00 = dot_grad(x0, y0, fx, fy)
        n10 = dot_grad(x1, y0, fx - 1, fy)
        n01 = dot_grad(x0, y1, fx, fy - 1)
        n11 = dot_grad(x1, y1, fx - 1, fy - 1)

        # Interpolate
        u = _fade(fx)
        v = _fade(fy)
        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        layer = _lerp(nx0, nx1, v)

        noise += layer * amplitude
        total_amplitude += amplitude
        amplitude *= 0.5
        scale *= 2.0

    # Normalize to [0, 1]
    noise /= total_amplitude
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
    return noise


def generate_terrain(
    width: int,
    height: int,
    scale: float = 0.07,
    octaves: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a terrain grid using Perlin noise.

    Returns
    -------
    np.ndarray of shape (width, height) with TerrainType int values.
    """
    noise = _perlin_2d(width, height, scale, octaves, seed)

    terrain = np.full((width, height), TerrainType.LAND, dtype=np.int8)

    # Threshold-based biome assignment (tuned for ~15% water, ~35% land, ~30% forest, ~20% mountain)
    terrain[noise < 0.15] = TerrainType.WATER
    terrain[(noise >= 0.15) & (noise < 0.50)] = TerrainType.LAND
    terrain[(noise >= 0.50) & (noise < 0.80)] = TerrainType.FOREST
    terrain[noise >= 0.80] = TerrainType.MOUNTAIN

    return terrain


def seed_resources(
    terrain: np.ndarray,
    max_food: float,
    max_materials: float,
    food_density: float,
    material_density: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Seed initial food and material grids based on terrain.

    Returns
    -------
    (food_grid, material_grid) — each shape (width, height)
    """
    rng = np.random.RandomState(seed + 1)
    w, h = terrain.shape

    food = np.zeros((w, h), dtype=np.float64)
    materials = np.zeros((w, h), dtype=np.float64)

    for tt in TerrainType:
        mask = terrain == tt
        food_mult = TERRAIN_FOOD_YIELD[tt]
        mat_mult = TERRAIN_MATERIAL_YIELD[tt]

        # Base level + small random jitter
        if food_mult > 0:
            base = max_food * food_density * food_mult
            food[mask] = base * rng.uniform(0.7, 1.0, size=mask.sum())
        if mat_mult > 0:
            base = max_materials * material_density * mat_mult
            materials[mask] = base * rng.uniform(0.7, 1.0, size=mask.sum())

    return food, materials
