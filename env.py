"""
env.py — Convenience re-exports for the simulation environment.

All core types are defined in their respective modules:
    config.py      → WorldConfig
    agent.py       → Agent, AgentClass, ClassTraits, CLASS_TRAITS
    terrain.py     → TerrainType, generate_terrain, seed_resources
    actions.py     → Action, MOVE_DELTAS, CLASS_ACTIONS
    world.py       → World
    observations.py → build_observation
    rewards.py     → compute_reward
    policies.py         → Policy, PolicyPool
    policy_networks.py  → NeuralNetPolicy, TabularPolicy, RuleBasedPolicy
    renderer.py         → Renderer
    analytics.py        → plot_metrics, plot_policy_fitness
"""

from config import WorldConfig
from agent import Agent, AgentClass, ClassTraits, CLASS_TRAITS
from terrain import TerrainType, generate_terrain, seed_resources
from actions import Action, MOVE_DELTAS, CLASS_ACTIONS
from world import World
from observations import build_observation
from rewards import compute_reward
from policies import Policy, PolicyPool
from policy_networks import NeuralNetPolicy, TabularPolicy, RuleBasedPolicy

__all__ = [
    "WorldConfig",
    "Agent", "AgentClass", "ClassTraits", "CLASS_TRAITS",
    "TerrainType", "generate_terrain", "seed_resources",
    "Action", "MOVE_DELTAS", "CLASS_ACTIONS",
    "World",
    "build_observation",
    "compute_reward",
    "Policy", "PolicyPool",
    "NeuralNetPolicy", "TabularPolicy", "RuleBasedPolicy",
]