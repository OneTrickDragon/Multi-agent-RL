"""
World — grid, agents, and the full simulation loop.

Simulation order each timestep:
    Observe → Act → Resolve → Update → Log
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import WorldConfig
from agent import Agent, AgentClass, CLASS_TRAITS
from terrain import (
    TerrainType,
    generate_terrain,
    seed_resources,
    TERRAIN_MOVE_COST,
    TERRAIN_FOOD_YIELD,
    TERRAIN_MATERIAL_YIELD,
    TERRAIN_DEFENSE_BONUS,
)
from actions import Action, MOVE_DELTAS, MOVE_ACTIONS, sanitize_action
from observations import build_observation
from rewards import compute_reward


class World:
    """
    Core simulation environment.

    Usage
    -----
    >>> cfg = WorldConfig()
    >>> world = World(cfg)
    >>> world.reset()
    >>> for _ in range(cfg.max_steps):
    ...     obs = world.observe()
    ...     actions = {a.id: policy(obs[a.id]) for a in world.alive_agents}
    ...     info = world.step(actions)
    ...     if info["done"]:
    ...         break
    """

    def __init__(self, config: WorldConfig):
        self.config = config
        self.terrain: np.ndarray = np.empty(0)
        self.food: np.ndarray = np.empty(0)
        self.materials: np.ndarray = np.empty(0)
        self.agents: List[Agent] = []
        self.current_step: int = 0
        self._next_agent_id: int = 0
        self._rng: np.random.RandomState = np.random.RandomState(config.terrain_seed)
        self.metrics: List[dict] = []

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """
        Regenerate terrain, resources, and agents.
        Returns initial observations for all agents.
        """
        cfg = self.config
        self._rng = np.random.RandomState(cfg.terrain_seed)
        self.current_step = 0
        self._next_agent_id = 0
        self.metrics = []

        # Terrain
        self.terrain = generate_terrain(
            cfg.width, cfg.height, cfg.terrain_scale, cfg.terrain_octaves, cfg.terrain_seed
        )

        # Resources
        self.food, self.materials = seed_resources(
            self.terrain,
            cfg.max_food_per_cell,
            cfg.max_materials_per_cell,
            cfg.initial_food_density,
            cfg.initial_material_density,
            seed=cfg.terrain_seed,
        )

        # Spawn agents
        self.agents = []
        self._spawn_initial_population()

        return self.observe()

    # ── Observe ────────────────────────────────────────────────────────────

    def observe(self) -> Dict[int, dict]:
        """Build observations for every alive agent."""
        alive = self.alive_agents
        total_alive = len(alive)
        return {
            a.id: build_observation(
                a, self.terrain, self.food, self.materials,
                alive, self.config, self.current_step, total_alive,
            )
            for a in alive
        }

    # ── Step ───────────────────────────────────────────────────────────────

    def step(self, actions: Dict[int, int]) -> dict:
        """
        Execute one full simulation timestep.

        Parameters
        ----------
        actions : mapping agent_id → int action

        Returns
        -------
        dict with keys:
            observations : per-agent observations
            rewards      : per-agent reward floats
            done         : bool — simulation ended
            info         : metrics snapshot
        """
        cfg = self.config
        alive = self.alive_agents

        # Sanitize actions
        agent_actions: Dict[int, Action] = {}
        for a in alive:
            raw = actions.get(a.id, 0)
            agent_actions[a.id] = sanitize_action(a.cls, raw)

        # ── Resolve all actions simultaneously ─────────────────────────
        resolution_info = self._resolve_actions(agent_actions)

        # ── Update world state ─────────────────────────────────────────
        self._update_resources()
        self._update_agents()

        # ── Check for dark-age reset ───────────────────────────────────
        alive_after = self.alive_agents
        if 0 < len(alive_after) < cfg.min_population:
            self._dark_age_repopulate()

        self.current_step += 1
        done = self.current_step >= cfg.max_steps or len(self.alive_agents) == 0

        # ── Compute rewards ────────────────────────────────────────────
        rewards: Dict[int, float] = {}
        total_alive = len(self.alive_agents)
        for a in self.alive_agents:
            info_a = resolution_info.get(a.id, {})
            rewards[a.id] = compute_reward(
                a, cfg, total_alive, **info_a,
            )

        # ── Log metrics ────────────────────────────────────────────────
        snapshot = self._snapshot()
        self.metrics.append(snapshot)

        return {
            "observations": self.observe(),
            "rewards": rewards,
            "done": done,
            "info": snapshot,
        }

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.alive]

    # ── Agent Spawning ─────────────────────────────────────────────────────

    def _spawn_initial_population(self) -> None:
        cfg = self.config
        passable = list(zip(*np.where(self.terrain != TerrainType.WATER)))
        self._rng.shuffle(passable)

        # Determine per-class counts
        class_order = [AgentClass.FARMER, AgentClass.WARRIOR, AgentClass.TRADER, AgentClass.LEADER]
        counts = {}
        remaining = cfg.initial_population
        for cls in class_order[:-1]:
            key = cls.name
            frac = cfg.class_distribution.get(key, 0.25)
            n = int(cfg.initial_population * frac)
            counts[cls] = n
            remaining -= n
        counts[class_order[-1]] = max(0, remaining)

        idx = 0
        for cls in class_order:
            for _ in range(counts[cls]):
                if idx >= len(passable):
                    break
                x, y = passable[idx]
                self._spawn_agent(x, y, cls)
                idx += 1

    def _spawn_agent(
        self, x: int, y: int, cls: AgentClass, policy_id: int = 0
    ) -> Agent:
        agent = Agent(
            id=self._next_agent_id,
            x=x, y=y,
            cls=cls,
            health=self.config.starting_health,
            food=self.config.starting_food,
            policy_id=policy_id,
        )
        self._next_agent_id += 1
        self.agents.append(agent)
        return agent

    # ── Action Resolution ──────────────────────────────────────────────────

    def _resolve_actions(self, actions: Dict[int, Action]) -> Dict[int, dict]:
        """
        Resolve all agent actions simultaneously. Returns per-agent info dicts
        for reward computation (food_harvested, damage_dealt, etc.).
        """
        cfg = self.config
        agent_map = {a.id: a for a in self.alive_agents}
        info: Dict[int, dict] = defaultdict(dict)

        # Build spatial index: (x, y) → list of agent ids
        cell_agents: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for a in self.alive_agents:
            cell_agents[(a.x, a.y)].append(a.id)

        # ── Movement ──────────────────────────────────────────────────
        move_requests: List[Tuple[int, int, int]] = []  # (agent_id, new_x, new_y)
        for aid, act in actions.items():
            if act in MOVE_ACTIONS:
                a = agent_map[aid]
                dx, dy = MOVE_DELTAS[act]
                nx, ny = a.x + dx, a.y + dy
                # Bounds check
                if 0 <= nx < cfg.width and 0 <= ny < cfg.height:
                    tt = TerrainType(self.terrain[nx, ny])
                    cost = TERRAIN_MOVE_COST[tt]
                    if cost > 0:  # passable
                        move_requests.append((aid, nx, ny))

        # Resolve movement conflicts (random priority)
        self._rng.shuffle(move_requests)
        for aid, nx, ny in move_requests:
            a = agent_map[aid]
            # Remove from old cell
            old_cell = cell_agents[(a.x, a.y)]
            if aid in old_cell:
                old_cell.remove(aid)
            # Move
            a.x, a.y = nx, ny
            cell_agents[(nx, ny)].append(aid)

        # ── Harvest ───────────────────────────────────────────────────
        for aid, act in actions.items():
            if act == Action.HARVEST:
                a = agent_map[aid]
                tt = TerrainType(self.terrain[a.x, a.y])

                # Food
                food_yield = TERRAIN_FOOD_YIELD[tt] * a.traits.harvest_yield * a.buff_multiplier
                harvested_food = min(self.food[a.x, a.y], food_yield)
                self.food[a.x, a.y] -= harvested_food
                a.food += harvested_food

                # Materials
                mat_yield = TERRAIN_MATERIAL_YIELD.get(tt, 0) * a.traits.material_yield * a.buff_multiplier
                max_carry = cfg.max_inventory + a.effective_max_inventory
                room = max(0.0, max_carry - a.materials)
                harvested_mat = min(self.materials[a.x, a.y], mat_yield, room)
                self.materials[a.x, a.y] -= harvested_mat
                a.materials += harvested_mat

                info[aid]["food_harvested"] = harvested_food
                info[aid]["materials_harvested"] = harvested_mat

        # ── Combat ────────────────────────────────────────────────────
        for aid, act in actions.items():
            if act == Action.ATTACK:
                attacker = agent_map[aid]
                # Find a target in the same cell (not self)
                targets = [
                    agent_map[tid] for tid in cell_agents[(attacker.x, attacker.y)]
                    if tid != aid and agent_map[tid].alive
                ]
                if not targets:
                    continue
                # Pick random target
                target = targets[self._rng.randint(len(targets))]

                # Calculate damage
                atk = cfg.base_attack * attacker.traits.attack * attacker.buff_multiplier
                dfn = cfg.base_defense * target.traits.defense * target.buff_multiplier
                tt = TerrainType(self.terrain[target.x, target.y])
                dfn *= (1.0 + TERRAIN_DEFENSE_BONUS[tt])

                randomness = 1.0 + self._rng.uniform(-cfg.combat_randomness, cfg.combat_randomness)
                damage = max(0.0, (atk - dfn) * randomness)
                target.health -= damage

                info[aid]["damage_dealt"] = info[aid].get("damage_dealt", 0.0) + damage

                if target.health <= 0:
                    target.alive = False
                    info[aid]["kills"] = info[aid].get("kills", 0) + 1
                    # Loot
                    loot = target.materials * attacker.traits.combat_loot
                    attacker.materials += loot
                    target.materials = 0.0

        # ── Trade ─────────────────────────────────────────────────────
        # Both parties must submit TRADE on the same tick while adjacent
        trade_requesters: List[int] = [
            aid for aid, act in actions.items() if act == Action.TRADE
        ]
        paired = set()
        for aid in trade_requesters:
            if aid in paired:
                continue
            a = agent_map[aid]
            # Look for another trader in the same cell
            for oid in cell_agents[(a.x, a.y)]:
                if oid == aid or oid in paired:
                    continue
                if actions.get(oid) == Action.TRADE:
                    o = agent_map[oid]
                    # Exchange food/materials
                    amount = cfg.trade_amount
                    eff_a = a.traits.trade_efficiency * a.buff_multiplier
                    eff_o = o.traits.trade_efficiency * o.buff_multiplier

                    # Simple exchange: each gives some food, gets materials (or vice versa)
                    food_swap = min(a.food, amount * eff_a)
                    mat_swap = min(o.materials, amount * eff_o)
                    a.food -= food_swap
                    o.food += food_swap
                    o.materials -= mat_swap
                    a.materials += mat_swap

                    info[aid]["trade_success"] = True
                    info[oid]["trade_success"] = True
                    paired.add(aid)
                    paired.add(oid)
                    break

        # ── Leader Rally ──────────────────────────────────────────────
        for aid, act in actions.items():
            if act == Action.RALLY:
                leader = agent_map[aid]
                radius = cfg.leader_rally_radius
                buffed = 0
                for other in self.alive_agents:
                    if other.id == aid:
                        continue
                    dx = abs(other.x - leader.x)
                    dy = abs(other.y - leader.y)
                    if dx <= radius and dy <= radius:
                        other.apply_buff(cfg.leader_buff_multiplier, cfg.leader_buff_duration)
                        buffed += 1
                info[aid]["allies_buffed"] = buffed

        # ── Reproduction ──────────────────────────────────────────────
        reproduce_requesters = [
            aid for aid, act in actions.items() if act == Action.REPRODUCE
        ]
        repro_paired = set()
        for aid in reproduce_requesters:
            if aid in repro_paired:
                continue
            a = agent_map[aid]
            if a.age < cfg.reproduction_age:
                continue
            if a.health < cfg.reproduction_health_min:
                continue
            if a.food < cfg.reproduction_food_cost:
                continue
            # Find partner in same cell
            for oid in cell_agents[(a.x, a.y)]:
                if oid == aid or oid in repro_paired:
                    continue
                if actions.get(oid) != Action.REPRODUCE:
                    continue
                o = agent_map[oid]
                if o.age < cfg.reproduction_age:
                    continue
                if o.health < cfg.reproduction_health_min:
                    continue
                if o.food < cfg.reproduction_food_cost:
                    continue

                # Reproduce
                a.food -= cfg.reproduction_food_cost
                o.food -= cfg.reproduction_food_cost
                # Child inherits random parent's class
                child_cls = a.cls if self._rng.random() < 0.5 else o.cls
                child_policy = a.policy_id if self._rng.random() < 0.5 else o.policy_id
                self._spawn_agent(a.x, a.y, child_cls, child_policy)
                repro_paired.add(aid)
                repro_paired.add(oid)
                break

        return dict(info)

    # ── Resource Regeneration ──────────────────────────────────────────────

    def _update_resources(self) -> None:
        cfg = self.config
        # Food regen (terrain-dependent)
        for tt in TerrainType:
            mask = self.terrain == tt
            yield_mult = TERRAIN_FOOD_YIELD[tt]
            if yield_mult > 0:
                self.food[mask] = np.minimum(
                    self.food[mask] + cfg.resource_regen_rate * yield_mult,
                    cfg.max_food_per_cell * yield_mult,
                )

        # Material regen
        for tt in TerrainType:
            mask = self.terrain == tt
            yield_mult = TERRAIN_MATERIAL_YIELD.get(tt, 0)
            if yield_mult > 0:
                self.materials[mask] = np.minimum(
                    self.materials[mask] + cfg.resource_regen_rate * 0.5 * yield_mult,
                    cfg.max_materials_per_cell * yield_mult,
                )

    # ── Agent Update ───────────────────────────────────────────────────────

    def _update_agents(self) -> None:
        cfg = self.config
        for a in self.alive_agents:
            # Age
            a.age += 1

            # Food drain
            drain = cfg.food_drain_per_tick * a.traits.food_consumption
            a.food -= drain

            # Starvation
            if a.food <= 0:
                a.food = 0.0
                a.health -= cfg.starvation_damage

            # Death check
            if a.health <= 0:
                a.alive = False

            # Tick buffs
            a.tick_buff()

    # ── Dark Age ───────────────────────────────────────────────────────────

    def _dark_age_repopulate(self) -> None:
        """
        When population falls below threshold, spawn new agents
        to bring it back to min_population.
        """
        cfg = self.config
        alive = self.alive_agents
        deficit = cfg.min_population - len(alive)
        if deficit <= 0:
            return

        passable = list(zip(*np.where(self.terrain != TerrainType.WATER)))
        self._rng.shuffle(passable)
        classes = list(AgentClass)
        for i in range(min(deficit, len(passable))):
            x, y = passable[i]
            cls = classes[self._rng.randint(len(classes))]
            self._spawn_agent(x, y, cls)

    # ── Metrics Snapshot ───────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        alive = self.alive_agents
        class_counts = {cls.name: 0 for cls in AgentClass}
        total_food = 0.0
        total_health = 0.0
        for a in alive:
            class_counts[a.cls.name] += 1
            total_food += a.food
            total_health += a.health

        n = max(len(alive), 1)
        return {
            "step": self.current_step,
            "alive": len(alive),
            "class_counts": class_counts,
            "avg_food": total_food / n,
            "avg_health": total_health / n,
            "total_agents_ever": self._next_agent_id,
        }
