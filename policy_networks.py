"""
Policy representations for the multi-agent RL simulation.

Three concrete strategies, all sharing a common base interface:

    1. NeuralNetPolicy  — Small MLP that maps flattened observations to
                          Q-values over the action space.  Trained online
                          via a simple replay-buffer + MSE TD update.
    2. TabularPolicy    — Classic Q-table over discretised observations.
                          Useful as a lightweight baseline.
    3. RuleBasedPolicy  — Hand-crafted heuristics per agent class.
                          Deterministic (no learning), good for sanity
                          checking the environment.

All policies expose:
    select_action(obs, cls, valid_actions, *, explore) -> int
    store_transition(obs, action, reward, next_obs, done)
    learn()                 # one gradient / table update
    state_dict() / load_state_dict()   # serialisation helpers
"""
from __future__ import annotations

import math
import copy
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agent import AgentClass
from actions import Action, CLASS_ACTIONS, NUM_ACTIONS


# ═══════════════════════════════════════════════════════════════════════════
#  Observation helpers
# ═══════════════════════════════════════════════════════════════════════════

def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten a structured observation dict into a single 1-D float32 vector."""
    parts = [
        obs["self"].ravel(),
        obs["local_terrain"].ravel(),
        obs["local_food"].ravel(),
        obs["local_materials"].ravel(),
        obs["local_agents"].ravel(),
        obs["global"].ravel(),
    ]
    return np.concatenate(parts).astype(np.float32)


def obs_dim(observation_radius: int = 3) -> int:
    """Return the length of a flattened observation vector."""
    w = 2 * observation_radius + 1          # 7 when radius=3
    self_len = 6
    terrain = w * w                         # 49
    food = w * w                            # 49
    materials = w * w                       # 49
    agents = w * w * 3                      # 147
    global_len = 2
    return self_len + terrain + food + materials + agents + global_len  # 302


# ═══════════════════════════════════════════════════════════════════════════
#  Replay buffer (shared by learning policies)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size ring buffer of transitions."""

    def __init__(self, capacity: int = 10_000):
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ═══════════════════════════════════════════════════════════════════════════
#  Base policy interface
# ═══════════════════════════════════════════════════════════════════════════

class BasePolicy(ABC):
    """Abstract base shared by all policy representations."""

    def __init__(self, policy_id: int):
        self.policy_id = policy_id
        self.fitness: float = 0.0
        self.num_agents: int = 0

    # -- core API ----------------------------------------------------------

    @abstractmethod
    def select_action(
        self,
        obs: dict,
        cls: AgentClass,
        valid_actions: Set[Action],
        *,
        explore: bool = True,
    ) -> int:
        """Return an action int given an observation and agent class."""

    def store_transition(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        done: bool,
    ) -> None:
        """Store a (s,a,r,s',d) tuple. Default: no-op for non-learning policies."""

    def learn(self) -> None:
        """Run one learning update. Default: no-op."""

    # -- fitness tracking (unchanged from original Policy) -----------------

    def update_fitness(self, reward: float) -> None:
        self.fitness = 0.95 * self.fitness + 0.05 * reward

    # -- serialisation stubs -----------------------------------------------

    def state_dict(self) -> dict:
        return {"policy_id": self.policy_id, "fitness": self.fitness}

    def load_state_dict(self, d: dict) -> None:
        self.fitness = d.get("fitness", 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Neural-network policy  (NumPy-only MLP, no PyTorch dependency)
# ═══════════════════════════════════════════════════════════════════════════

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


class _Linear:
    """Single dense layer with Xavier init."""

    def __init__(self, in_features: int, out_features: int):
        scale = math.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros(out_features, dtype=np.float32)
        # Adam moment buffers
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)


class NeuralNetPolicy(BasePolicy):
    """
    Two-hidden-layer MLP mapping flattened observations to Q-values.

    Architecture
    ------------
        flatten(obs) → Linear(obs_dim, 128) → ReLU
                     → Linear(128, 64)       → ReLU
                     → Linear(64, NUM_ACTIONS)

    Training: simple online DQN with replay buffer and epsilon-greedy.
    Uses raw NumPy so there is zero external dependency beyond numpy.
    """

    def __init__(
        self,
        policy_id: int,
        input_dim: int = 302,
        hidden1: int = 128,
        hidden2: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        learn_every: int = 4,
        target_update_every: int = 200,
    ):
        super().__init__(policy_id)
        self.input_dim = input_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.target_update_every = target_update_every

        # Online network
        self.fc1 = _Linear(input_dim, hidden1)
        self.fc2 = _Linear(hidden1, hidden2)
        self.fc3 = _Linear(hidden2, NUM_ACTIONS)

        # Target network (deep copy)
        self.t_fc1 = copy.deepcopy(self.fc1)
        self.t_fc2 = copy.deepcopy(self.fc2)
        self.t_fc3 = copy.deepcopy(self.fc3)

        self.buffer = ReplayBuffer(buffer_capacity)
        self._step_count: int = 0
        self._adam_t: int = 0  # Adam timestep

    # -- forward pass ------------------------------------------------------

    @staticmethod
    def _forward(x: np.ndarray, fc1: _Linear, fc2: _Linear, fc3: _Linear):
        """Return (q_values, pre-activations for backprop)."""
        z1 = x @ fc1.W + fc1.b
        a1 = _relu(z1)
        z2 = a1 @ fc2.W + fc2.b
        a2 = _relu(z2)
        q = a2 @ fc3.W + fc3.b
        return q, (x, z1, a1, z2, a2)

    def _q_values(self, obs_flat: np.ndarray) -> np.ndarray:
        q, _ = self._forward(obs_flat, self.fc1, self.fc2, self.fc3)
        return q

    def _target_q(self, obs_flat: np.ndarray) -> np.ndarray:
        q, _ = self._forward(obs_flat, self.t_fc1, self.t_fc2, self.t_fc3)
        return q

    # -- action selection --------------------------------------------------

    def select_action(
        self,
        obs: dict,
        cls: AgentClass,
        valid_actions: Set[Action] | None = None,
        *,
        explore: bool = True,
    ) -> int:
        if valid_actions is None:
            valid_actions = CLASS_ACTIONS[cls]
        valid_list = list(valid_actions)

        # Epsilon-greedy
        if explore and random.random() < self.epsilon:
            return int(valid_list[random.randrange(len(valid_list))])

        flat = flatten_obs(obs)
        q = self._q_values(flat)
        # Mask invalid actions to -inf
        mask = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
        for a in valid_list:
            mask[int(a)] = 0.0
        q_masked = q + mask
        return int(np.argmax(q_masked))

    # -- experience storage ------------------------------------------------

    def store_transition(self, obs, action, reward, next_obs, done):
        self.buffer.push(Transition(
            flatten_obs(obs), action, reward,
            flatten_obs(next_obs) if next_obs is not None else np.zeros(self.input_dim, dtype=np.float32),
            done,
        ))
        self._step_count += 1
        # Decay epsilon
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self._step_count / max(self.epsilon_decay, 1))

    # -- learning (DQN with Adam) ------------------------------------------

    def learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        if self._step_count % self.learn_every != 0:
            return

        batch = self.buffer.sample(self.batch_size)
        obs_b = np.stack([t.obs for t in batch])
        act_b = np.array([t.action for t in batch], dtype=np.int64)
        rew_b = np.array([t.reward for t in batch], dtype=np.float32)
        nobs_b = np.stack([t.next_obs for t in batch])
        done_b = np.array([t.done for t in batch], dtype=np.float32)

        # Current Q
        q_all, cache = self._forward(obs_b, self.fc1, self.fc2, self.fc3)
        x_in, z1, a1, z2, a2 = cache
        q_pred = q_all[np.arange(self.batch_size), act_b]

        # Target Q
        with np.errstate(invalid="ignore"):
            q_next = self._target_q(nobs_b)
        q_target = rew_b + self.gamma * np.max(q_next, axis=1) * (1.0 - done_b)

        # TD error
        td = q_pred - q_target  # (B,)

        # -- Backprop (manual) -------------------------------------------
        # dL/dq_all: one-hot on the chosen action
        dq = np.zeros_like(q_all)
        dq[np.arange(self.batch_size), act_b] = td / self.batch_size

        # fc3
        dW3 = a2.T @ dq
        db3 = dq.sum(axis=0)
        da2 = dq @ self.fc3.W.T

        # ReLU after fc2
        da2 = da2 * _relu_grad(z2)

        # fc2
        dW2 = a1.T @ da2
        db2 = da2.sum(axis=0)
        da1 = da2 @ self.fc2.W.T

        # ReLU after fc1
        da1 = da1 * _relu_grad(z1)

        # fc1
        dW1 = x_in.T @ da1
        db1 = da1.sum(axis=0)

        # Adam update
        self._adam_t += 1
        for layer, gW, gb in [
            (self.fc1, dW1, db1),
            (self.fc2, dW2, db2),
            (self.fc3, dW3, db3),
        ]:
            self._adam_step(layer, gW, gb)

        # Periodically sync target network
        if self._step_count % self.target_update_every == 0:
            self.t_fc1 = copy.deepcopy(self.fc1)
            self.t_fc2 = copy.deepcopy(self.fc2)
            self.t_fc3 = copy.deepcopy(self.fc3)

    def _adam_step(self, layer: _Linear, gW: np.ndarray, gb: np.ndarray):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = self._adam_t

        layer.mW = beta1 * layer.mW + (1 - beta1) * gW
        layer.vW = beta2 * layer.vW + (1 - beta2) * gW ** 2
        mW_hat = layer.mW / (1 - beta1 ** t)
        vW_hat = layer.vW / (1 - beta2 ** t)
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

        layer.mb = beta1 * layer.mb + (1 - beta1) * gb
        layer.vb = beta2 * layer.vb + (1 - beta2) * gb ** 2
        mb_hat = layer.mb / (1 - beta1 ** t)
        vb_hat = layer.vb / (1 - beta2 ** t)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # -- serialisation -----------------------------------------------------

    def state_dict(self) -> dict:
        d = super().state_dict()
        d.update({
            "fc1_W": self.fc1.W.copy(), "fc1_b": self.fc1.b.copy(),
            "fc2_W": self.fc2.W.copy(), "fc2_b": self.fc2.b.copy(),
            "fc3_W": self.fc3.W.copy(), "fc3_b": self.fc3.b.copy(),
            "epsilon": self.epsilon, "step_count": self._step_count,
        })
        return d

    def load_state_dict(self, d: dict) -> None:
        super().load_state_dict(d)
        self.fc1.W = d["fc1_W"]; self.fc1.b = d["fc1_b"]
        self.fc2.W = d["fc2_W"]; self.fc2.b = d["fc2_b"]
        self.fc3.W = d["fc3_W"]; self.fc3.b = d["fc3_b"]
        self.epsilon = d.get("epsilon", self.epsilon_end)
        self._step_count = d.get("step_count", 0)
        # Sync target
        self.t_fc1 = copy.deepcopy(self.fc1)
        self.t_fc2 = copy.deepcopy(self.fc2)
        self.t_fc3 = copy.deepcopy(self.fc3)

    def mutate(self, sigma: float = 0.02) -> "NeuralNetPolicy":
        """Return a new policy whose weights are this policy's + Gaussian noise."""
        child = copy.deepcopy(self)
        child.policy_id = -1  # caller must assign
        for layer in (child.fc1, child.fc2, child.fc3):
            layer.W += np.random.randn(*layer.W.shape).astype(np.float32) * sigma
            layer.b += np.random.randn(*layer.b.shape).astype(np.float32) * sigma
        # Reset target to mutated online
        child.t_fc1 = copy.deepcopy(child.fc1)
        child.t_fc2 = copy.deepcopy(child.fc2)
        child.t_fc3 = copy.deepcopy(child.fc3)
        return child


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Tabular Q-learning policy
# ═══════════════════════════════════════════════════════════════════════════

def _discretise_obs(obs: dict, n_bins: int = 4) -> tuple:
    """
    Hash a structured observation into a discrete state key.

    We keep it coarse (few bins) so the table stays tractable:
      - self state: each feature binned into n_bins buckets
      - local food/materials: total in view binned
      - nearby agent count
      - global signals binned
    """
    s = obs["self"]
    bins = np.linspace(0, 1, n_bins + 1)[1:-1]  # interior edges

    key_parts: list = []
    # Self features (skip class index, keep health/food/materials fracs)
    key_parts.append(int(s[0]))                              # class
    key_parts.append(int(np.digitize(np.clip(s[1], 0, 1), bins)))  # health frac
    key_parts.append(int(np.digitize(np.clip(s[2], 0, 1), bins)))  # food frac
    key_parts.append(int(np.digitize(np.clip(s[3], 0, 1), bins)))  # material frac

    # Summarise spatial channels
    total_food = float(obs["local_food"].sum())
    total_mat = float(obs["local_materials"].sum())
    n_agents = int((obs["local_agents"][:, :, 2] > 0).sum())

    food_bins = np.array([2.0, 8.0, 20.0])
    mat_bins = np.array([2.0, 6.0, 15.0])
    agent_bins = np.array([1, 3, 6])

    key_parts.append(int(np.digitize(total_food, food_bins)))
    key_parts.append(int(np.digitize(total_mat, mat_bins)))
    key_parts.append(int(np.digitize(n_agents, agent_bins)))

    return tuple(key_parts)


class TabularPolicy(BasePolicy):
    """
    Q-table policy with epsilon-greedy exploration.

    State space is discretised via `_discretise_obs` to keep
    the table manageable.
    """

    def __init__(
        self,
        policy_id: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
    ):
        super().__init__(policy_id)
        self.q: Dict[tuple, np.ndarray] = {}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._step_count = 0
        self._pending: Optional[Tuple[tuple, int, float, tuple, bool]] = None

    def _get_q(self, key: tuple) -> np.ndarray:
        if key not in self.q:
            self.q[key] = np.zeros(NUM_ACTIONS, dtype=np.float32)
        return self.q[key]

    def select_action(self, obs, cls, valid_actions=None, *, explore=True):
        if valid_actions is None:
            valid_actions = CLASS_ACTIONS[cls]
        valid_list = list(valid_actions)

        if explore and random.random() < self.epsilon:
            return int(valid_list[random.randrange(len(valid_list))])

        key = _discretise_obs(obs)
        q = self._get_q(key)
        mask = np.full(NUM_ACTIONS, -np.inf)
        for a in valid_list:
            mask[int(a)] = 0.0
        return int(np.argmax(q + mask))

    def store_transition(self, obs, action, reward, next_obs, done):
        key = _discretise_obs(obs)
        nkey = _discretise_obs(next_obs) if next_obs is not None else key
        self._pending = (key, action, reward, nkey, done)
        self._step_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self._step_count / max(self.epsilon_decay, 1))

    def learn(self):
        if self._pending is None:
            return
        key, action, reward, nkey, done = self._pending
        self._pending = None
        q = self._get_q(key)
        nq = self._get_q(nkey)
        target = reward + (0.0 if done else self.gamma * np.max(nq))
        q[action] += self.lr * (target - q[action])

    def state_dict(self):
        d = super().state_dict()
        d["q_table"] = {str(k): v.tolist() for k, v in self.q.items()}
        d["epsilon"] = self.epsilon
        d["step_count"] = self._step_count
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Rule-based (heuristic) policy
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedPolicy(BasePolicy):
    """
    Deterministic heuristic policy — no learning.

    Class-specific behaviour:
      FARMER  : move toward highest-food cell in view; harvest when on food
      WARRIOR : move toward nearest enemy; attack when adjacent
      TRADER  : move toward nearest other agent; trade when co-located
      LEADER  : move toward cluster of allies; rally when allies nearby
    """

    def select_action(self, obs, cls, valid_actions=None, *, explore=True):
        if valid_actions is None:
            valid_actions = CLASS_ACTIONS[cls]

        r = obs["local_food"].shape[0] // 2  # observation radius
        agents_grid = obs["local_agents"]     # (W, W, 3)

        if cls == AgentClass.FARMER:
            return self._farmer(obs, r, valid_actions)
        elif cls == AgentClass.WARRIOR:
            return self._warrior(obs, r, valid_actions)
        elif cls == AgentClass.TRADER:
            return self._trader(obs, r, valid_actions)
        elif cls == AgentClass.LEADER:
            return self._leader(obs, r, valid_actions)
        return int(Action.NOOP)

    # -- per-class heuristics ---------------------------------------------

    @staticmethod
    def _farmer(obs, r, valid):
        # If standing on food, harvest
        center_food = obs["local_food"][r, r]
        if center_food > 0.5 and Action.HARVEST in valid:
            return int(Action.HARVEST)
        # Move toward richest food cell
        return _move_toward_max(obs["local_food"], r, valid)

    @staticmethod
    def _warrior(obs, r, valid):
        agents = obs["local_agents"]
        # If enemy in same cell, attack
        if agents[r, r, 2] > 0 and Action.ATTACK in valid:
            return int(Action.ATTACK)
        # Move toward nearest agent
        return _move_toward_agent(agents, r, valid)

    @staticmethod
    def _trader(obs, r, valid):
        agents = obs["local_agents"]
        if agents[r, r, 2] > 0 and Action.TRADE in valid:
            return int(Action.TRADE)
        return _move_toward_agent(agents, r, valid)

    @staticmethod
    def _leader(obs, r, valid):
        agents = obs["local_agents"]
        # Count allies in rally radius
        ally_count = int((agents[:, :, 2] > 0).sum())
        if ally_count >= 2 and Action.RALLY in valid:
            return int(Action.RALLY)
        return _move_toward_agent(agents, r, valid)

    # No learning
    def store_transition(self, *a, **kw): pass
    def learn(self): pass


# ── heuristic helpers ────────────────────────────────────────────────────

def _move_toward_max(grid: np.ndarray, r: int, valid: set) -> int:
    """Move one step toward the cell with the highest value in `grid`."""
    best = np.unravel_index(np.argmax(grid), grid.shape)
    return _direction_to(best[0] - r, best[1] - r, valid)


def _move_toward_agent(agents: np.ndarray, r: int, valid: set) -> int:
    """Move toward nearest present agent in the local_agents channel."""
    presence = agents[:, :, 2]
    coords = np.argwhere(presence > 0)
    if len(coords) == 0:
        # No agents visible — random move
        moves = [a for a in valid if a in {Action.MOVE_UP, Action.MOVE_DOWN,
                                            Action.MOVE_LEFT, Action.MOVE_RIGHT}]
        return int(moves[random.randrange(len(moves))]) if moves else int(Action.NOOP)
    # Find closest
    dists = np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - r)
    nearest = coords[np.argmin(dists)]
    return _direction_to(nearest[0] - r, nearest[1] - r, valid)


def _direction_to(dx: int, dy: int, valid: set) -> int:
    """Convert a relative (dx, dy) into the best single-step move action."""
    candidates = []
    if dx < 0 and Action.MOVE_UP in valid:
        candidates.append((abs(dx), Action.MOVE_UP))
    if dx > 0 and Action.MOVE_DOWN in valid:
        candidates.append((abs(dx), Action.MOVE_DOWN))
    if dy < 0 and Action.MOVE_LEFT in valid:
        candidates.append((abs(dy), Action.MOVE_LEFT))
    if dy > 0 and Action.MOVE_RIGHT in valid:
        candidates.append((abs(dy), Action.MOVE_RIGHT))
    if not candidates:
        return int(Action.NOOP)
    # Move along the axis with the larger offset
    candidates.sort(key=lambda c: c[0], reverse=True)
    return int(candidates[0][1])
