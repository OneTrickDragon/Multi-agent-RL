"""
Agent dataclass, AgentClass enum, and class trait multiplier tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict


class AgentClass(IntEnum):
    """Four agent archetypes, each with distinct mechanical roles."""
    FARMER = 0
    WARRIOR = 1
    TRADER = 2
    LEADER = 3


# ── Class Trait Tables ─────────────────────────────────────────────────────
# Each trait is a multiplier applied on top of the base value from WorldConfig.

@dataclass(frozen=True)
class ClassTraits:
    """Immutable trait multipliers for a given agent class."""
    harvest_yield: float = 1.0       # Multiplier on food gathered per harvest action
    material_yield: float = 1.0      # Multiplier on materials gathered
    attack: float = 1.0              # Multiplier on base_attack
    defense: float = 1.0             # Multiplier on base_defense
    food_consumption: float = 1.0    # Multiplier on food_drain_per_tick
    move_cost: float = 1.0           # Multiplier on movement food cost
    inventory_bonus: int = 0         # Added to max_inventory
    trade_efficiency: float = 1.0    # Multiplier on trade exchange amounts
    combat_loot: float = 0.0         # Fraction of defeated enemy's inventory looted


CLASS_TRAITS: Dict[AgentClass, ClassTraits] = {
    AgentClass.FARMER: ClassTraits(
        harvest_yield=1.8,
        material_yield=1.3,
        attack=0.5,
        defense=0.7,
        food_consumption=0.8,
    ),
    AgentClass.WARRIOR: ClassTraits(
        attack=1.6,
        defense=1.4,
        food_consumption=1.3,
        combat_loot=0.3,
        harvest_yield=0.6,
    ),
    AgentClass.TRADER: ClassTraits(
        trade_efficiency=1.5,
        inventory_bonus=20,
        harvest_yield=0.9,
        attack=0.6,
        defense=0.8,
        move_cost=0.8,
    ),
    AgentClass.LEADER: ClassTraits(
        attack=0.8,
        defense=1.0,
        harvest_yield=0.7,
        food_consumption=0.9,
    ),
}


# ── Agent Dataclass ────────────────────────────────────────────────────────

@dataclass
class Agent:
    """
    Mutable agent state. The World owns all Agent instances;
    agents do not hold references back to the world.
    """
    id: int
    x: int
    y: int
    cls: AgentClass
    health: float
    food: float
    age: int = 0
    policy_id: int = 0
    materials: float = 0.0
    alive: bool = True

    # Transient buff state (from Leader rally)
    buff_multiplier: float = 1.0
    buff_ticks_remaining: int = 0

    @property
    def traits(self) -> ClassTraits:
        return CLASS_TRAITS[self.cls]

    @property
    def effective_max_inventory(self) -> int:
        """Max inventory including class bonus."""
        # Base max_inventory comes from WorldConfig at resolution time;
        # this property provides the class-specific offset.
        return self.traits.inventory_bonus

    def tick_buff(self) -> None:
        """Decay active buff by one tick."""
        if self.buff_ticks_remaining > 0:
            self.buff_ticks_remaining -= 1
            if self.buff_ticks_remaining == 0:
                self.buff_multiplier = 1.0

    def apply_buff(self, multiplier: float, duration: int) -> None:
        """Apply (or refresh) a leader rally buff."""
        self.buff_multiplier = multiplier
        self.buff_ticks_remaining = duration
