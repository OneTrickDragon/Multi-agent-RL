"""
Action definitions and validity checks.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Dict, Set

from agent import AgentClass


class Action(IntEnum):
    """All possible agent actions."""
    NOOP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    HARVEST = 5
    ATTACK = 6
    TRADE = 7
    RALLY = 8        # Leader only
    REPRODUCE = 9


# Movement deltas: (dx, dy)
MOVE_DELTAS: Dict[Action, tuple[int, int]] = {
    Action.MOVE_UP: (0, -1),
    Action.MOVE_DOWN: (0, 1),
    Action.MOVE_LEFT: (-1, 0),
    Action.MOVE_RIGHT: (1, 0),
}

MOVE_ACTIONS: Set[Action] = set(MOVE_DELTAS.keys())

# Actions available per class — prevents invalid class/action combos
CLASS_ACTIONS: Dict[AgentClass, Set[Action]] = {
    AgentClass.FARMER: {
        Action.NOOP, Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.HARVEST, Action.REPRODUCE,
    },
    AgentClass.WARRIOR: {
        Action.NOOP, Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.HARVEST, Action.ATTACK, Action.REPRODUCE,
    },
    AgentClass.TRADER: {
        Action.NOOP, Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.HARVEST, Action.TRADE, Action.REPRODUCE,
    },
    AgentClass.LEADER: {
        Action.NOOP, Action.MOVE_UP, Action.MOVE_DOWN,
        Action.MOVE_LEFT, Action.MOVE_RIGHT,
        Action.HARVEST, Action.RALLY, Action.REPRODUCE,
    },
}

NUM_ACTIONS = len(Action)


def is_valid_action(cls: AgentClass, action: Action) -> bool:
    """Check if a class may perform this action."""
    return action in CLASS_ACTIONS[cls]


def sanitize_action(cls: AgentClass, action: int) -> Action:
    """Clamp raw int to valid Action; fallback to NOOP if invalid for class."""
    try:
        act = Action(action)
    except ValueError:
        return Action.NOOP
    if act not in CLASS_ACTIONS[cls]:
        return Action.NOOP
    return act
