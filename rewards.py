"""
Layered reward computation.

Three reward signals combined per timestep:
  1. Survival reward  — small positive each tick alive
  2. Role reward      — class-specific performance bonus
  3. Population signal — mild bonus tied to total population
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from config import WorldConfig


# ── Weights ─────────────────────────────────────────────────────────────────
SURVIVAL_REWARD = 0.1
POPULATION_WEIGHT = 0.01          # Per-tick bonus = weight * (alive / initial)

# Role-specific reward rates
FARMER_HARVEST_REWARD = 0.3       # Per unit of food harvested
WARRIOR_KILL_REWARD = 2.0         # Per enemy defeated
WARRIOR_DAMAGE_REWARD = 0.05      # Per HP of damage dealt
TRADER_TRADE_REWARD = 1.0         # Per successful trade
LEADER_ALLY_ALIVE_REWARD = 0.02   # Per allied agent still alive in buff radius


def compute_reward(
    agent: Agent,
    config: WorldConfig,
    total_alive: int,
    *,
    food_harvested: float = 0.0,
    materials_harvested: float = 0.0,
    damage_dealt: float = 0.0,
    kills: int = 0,
    trade_success: bool = False,
    allies_buffed: int = 0,
) -> float:
    """
    Compute total reward for an agent this timestep.

    Keyword args track what happened during action resolution.
    """
    from agent import AgentClass

    reward = 0.0

    # 1. Survival
    reward += SURVIVAL_REWARD

    # 2. Role reward
    if agent.cls == AgentClass.FARMER:
        reward += FARMER_HARVEST_REWARD * food_harvested
        reward += FARMER_HARVEST_REWARD * 0.5 * materials_harvested  # some credit for materials too
    elif agent.cls == AgentClass.WARRIOR:
        reward += WARRIOR_KILL_REWARD * kills
        reward += WARRIOR_DAMAGE_REWARD * damage_dealt
    elif agent.cls == AgentClass.TRADER:
        reward += TRADER_TRADE_REWARD * int(trade_success)
    elif agent.cls == AgentClass.LEADER:
        reward += LEADER_ALLY_ALIVE_REWARD * allies_buffed

    # 3. Population signal
    if config.initial_population > 0:
        reward += POPULATION_WEIGHT * (total_alive / config.initial_population)

    return reward
