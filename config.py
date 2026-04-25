"""
WorldConfig — all tunable simulation constants in one place.
"""
from dataclasses import dataclass, field


@dataclass
class WorldConfig:
    # --- Map ---
    width: int = 64
    height: int = 64
    terrain_scale: float = 0.07          # Perlin noise frequency (lower = larger biomes)
    terrain_octaves: int = 4
    terrain_seed: int = 42

    # --- Population ---
    initial_population: int = 80         # Total agents at start
    class_distribution: dict = field(    # Fraction per class (must sum to 1.0)
        default_factory=lambda: {
            "FARMER": 0.35,
            "WARRIOR": 0.25,
            "TRADER": 0.20,
            "LEADER": 0.20,
        }
    )
    min_population: int = 10             # "Dark age" reset threshold

    # --- Agent defaults ---
    starting_health: float = 100.0
    starting_food: float = 50.0
    food_drain_per_tick: float = 1.0     # Food consumed per timestep
    starvation_damage: float = 5.0       # Health lost per tick when food == 0
    max_inventory: int = 50              # Soft cap on carried materials

    # --- Resources ---
    resource_regen_rate: float = 0.3     # Units regenerated per cell per tick
    max_food_per_cell: float = 10.0
    max_materials_per_cell: float = 8.0
    initial_food_density: float = 0.6    # Fraction of max at reset
    initial_material_density: float = 0.4

    # --- Observation ---
    observation_radius: int = 3          # 7x7 local window (radius 3 around agent)

    # --- Combat ---
    base_attack: float = 10.0
    base_defense: float = 5.0
    combat_randomness: float = 0.2       # ±20% stochastic modifier

    # --- Reproduction ---
    reproduction_age: int = 50           # Minimum age to reproduce
    reproduction_food_cost: float = 20.0 # Food spent by each parent
    reproduction_health_min: float = 60.0

    # --- Leader ---
    leader_rally_radius: int = 2
    leader_buff_multiplier: float = 1.15 # 15% buff
    leader_buff_duration: int = 5        # Ticks

    # --- Trade ---
    trade_amount: int = 5                # Units exchanged per trade action

    # --- Simulation ---
    max_steps: int = 5000

    # --- Renderer ---
    cell_size: int = 10                  # Pixels per grid cell
    fps: int = 30
    steps_per_frame: int = 1             # Simulation ticks per rendered frame
