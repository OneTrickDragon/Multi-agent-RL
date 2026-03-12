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


class World:
    def __init__(self, config: WorldConfig):
        self.terrain = np.zeros((config.width, config.height))   # terrain types
        self.food    = np.zeros((config.width, config.height))   # food per cell
        self.materials = np.zeros((config.width, config.height)) # materials per cell
        
    def reset(self): ...
    def step(self): ...
    def get_observation(self, agent: Agent): ...
    def _resolve_actions(self, actions: dict): ...
    def _update_resources(self): ...
    def _update_agents(self): ...
    def render(self): ...