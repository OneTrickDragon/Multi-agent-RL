"""
Policy pool system.

At this stage, policies are random action selectors.
The architecture is designed for later upgrades to neural-net policies
and evolutionary adaptation.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from agent import Agent, AgentClass
from actions import Action, CLASS_ACTIONS, NUM_ACTIONS


class Policy:
    """
    A callable policy mapping observation → action int.
    Base implementation: uniform random over valid class actions.
    """

    def __init__(self, policy_id: int):
        self.policy_id = policy_id
        self.fitness: float = 0.0       # Accumulated fitness for adaptation
        self.num_agents: int = 0        # How many agents currently use this policy

    def __call__(self, obs: dict, cls: AgentClass) -> int:
        """Select an action given an observation and agent class."""
        valid = list(CLASS_ACTIONS[cls])
        return int(valid[np.random.randint(len(valid))])

    def update_fitness(self, reward: float) -> None:
        """Running average of rewards from agents using this policy."""
        self.fitness = 0.95 * self.fitness + 0.05 * reward


class PolicyPool:
    """
    Shared pool of policies. Agents reference policies by ID.

    Parameters
    ----------
    size : number of policies in the pool
    """

    def __init__(self, size: int = 10):
        self.policies: Dict[int, Policy] = {
            i: Policy(i) for i in range(size)
        }
        self.size = size

    def get(self, policy_id: int) -> Policy:
        return self.policies[policy_id]

    def assign_random(self) -> int:
        """Return a random policy ID."""
        return int(np.random.randint(self.size))

    def select_actions(
        self, agents: List[Agent], observations: Dict[int, dict]
    ) -> Dict[int, int]:
        """
        Query each agent's assigned policy for an action.

        Returns mapping agent_id → action int.
        """
        actions = {}
        for a in agents:
            if not a.alive:
                continue
            obs = observations.get(a.id)
            if obs is None:
                actions[a.id] = int(Action.NOOP)
                continue
            policy = self.get(a.policy_id)
            actions[a.id] = policy(obs, a.cls)
        return actions

    def update_fitness(self, rewards: Dict[int, float], agents: List[Agent]) -> None:
        """
        Feed rewards back into policies to track fitness.
        """
        # Reset counts
        for p in self.policies.values():
            p.num_agents = 0

        for a in agents:
            if not a.alive:
                continue
            p = self.get(a.policy_id)
            p.num_agents += 1
            r = rewards.get(a.id, 0.0)
            p.update_fitness(r)

    def get_stats(self) -> List[dict]:
        """Return per-policy stats for analytics."""
        return [
            {
                "id": pid,
                "fitness": p.fitness,
                "num_agents": p.num_agents,
            }
            for pid, p in sorted(self.policies.items())
        ]
