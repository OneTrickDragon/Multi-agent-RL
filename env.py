from dataclasses import dataclass, field
import numpy as np

@dataclass
class WorldConfig:
    width: int
    height: int
    initial_population: int
    observation_radius: int
    resource_regen_rate: float
    max_resource_per_cell: float
    starting_food: int
    food_drain_per_tick: int

@dataclass
class Agent:
    id: int
    x: int
    y: int
    cls: AgentClass   
    health: float
    food: float
    age: int
    policy_id: int    