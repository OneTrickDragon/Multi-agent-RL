"""
Policy pool system.

Supports three pluggable policy representations:
    - "neural"     → NeuralNetPolicy   (MLP + DQN, online learning)
    - "tabular"    → TabularPolicy     (Q-table, online learning)
    - "rule_based" → RuleBasedPolicy   (deterministic heuristics, no learning)

The pool owns all policies.  Agents reference them by integer ID.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from agent import Agent, AgentClass
from actions import Action, CLASS_ACTIONS, NUM_ACTIONS
from policy_networks import (
    BasePolicy,
    NeuralNetPolicy,
    TabularPolicy,
    RuleBasedPolicy,
    obs_dim,
)

# Valid policy type strings
POLICY_TYPES = {"neural", "tabular", "rule_based"}


def _make_policy(policy_id: int, kind: str, input_dim: int, **kw) -> BasePolicy:
    """Factory: instantiate the correct policy class."""
    if kind == "neural":
        return NeuralNetPolicy(policy_id, input_dim=input_dim, **kw)
    elif kind == "tabular":
        return TabularPolicy(policy_id, **kw)
    elif kind == "rule_based":
        return RuleBasedPolicy(policy_id)
    raise ValueError(f"Unknown policy type: {kind!r}  (expected one of {POLICY_TYPES})")


# Legacy shim so old imports still work
Policy = BasePolicy


class PolicyPool:
    """
    Shared pool of policies.  Agents reference policies by ID.

    Parameters
    ----------
    size : number of policies in the pool
    kind : one of "neural", "tabular", "rule_based"
    observation_radius : used to compute input_dim for neural policies
    """

    def __init__(
        self,
        size: int = 10,
        kind: str = "neural",
        observation_radius: int = 3,
        **policy_kwargs,
    ):
        if kind not in POLICY_TYPES:
            raise ValueError(f"Unknown policy type {kind!r}")
        self.kind = kind
        self.size = size
        self._input_dim = obs_dim(observation_radius)
        self.policies: Dict[int, BasePolicy] = {
            i: _make_policy(i, kind, self._input_dim, **policy_kwargs)
            for i in range(size)
        }

    def get(self, policy_id: int) -> BasePolicy:
        return self.policies[policy_id]

    def assign_random(self) -> int:
        """Return a random policy ID."""
        return int(np.random.randint(self.size))

    # ── Action selection ──────────────────────────────────────────────────

    def select_actions(
        self,
        agents: List[Agent],
        observations: Dict[int, dict],
        *,
        explore: bool = True,
    ) -> Dict[int, int]:
        """
        Query each agent's assigned policy for an action.

        Returns mapping agent_id → action int.
        """
        actions: Dict[int, int] = {}
        for a in agents:
            if not a.alive:
                continue
            obs = observations.get(a.id)
            if obs is None:
                actions[a.id] = int(Action.NOOP)
                continue
            policy = self.get(a.policy_id)
            valid = CLASS_ACTIONS[a.cls]
            actions[a.id] = policy.select_action(obs, a.cls, valid, explore=explore)
        return actions

    # ── Experience + learning ─────────────────────────────────────────────

    def store_transitions(
        self,
        agents: List[Agent],
        prev_obs: Dict[int, dict],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        next_obs: Dict[int, dict],
        done: bool,
    ) -> None:
        """Push (s,a,r,s',d) for every alive agent into its policy's buffer."""
        for a in agents:
            if not a.alive:
                continue
            po = prev_obs.get(a.id)
            no = next_obs.get(a.id)
            act = actions.get(a.id)
            rew = rewards.get(a.id, 0.0)
            if po is None or act is None:
                continue
            self.get(a.policy_id).store_transition(po, act, rew, no, done)

    def learn_all(self) -> None:
        """Run one learning step on every policy in the pool."""
        for p in self.policies.values():
            p.learn()

    # ── Fitness tracking (unchanged) ──────────────────────────────────────

    def update_fitness(self, rewards: Dict[int, float], agents: List[Agent]) -> None:
        for p in self.policies.values():
            p.num_agents = 0
        for a in agents:
            if not a.alive:
                continue
            p = self.get(a.policy_id)
            p.num_agents += 1
            r = rewards.get(a.id, 0.0)
            p.update_fitness(r)

    # ── Stats / analytics ─────────────────────────────────────────────────

    def get_stats(self) -> List[dict]:
        """Return per-policy stats for analytics."""
        return [
            {
                "id": pid,
                "fitness": p.fitness,
                "num_agents": p.num_agents,
                "type": type(p).__name__,
            }
            for pid, p in sorted(self.policies.items())
        ]
