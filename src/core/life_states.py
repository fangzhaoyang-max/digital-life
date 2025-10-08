"""
Core life states and enums for the Digital Life system
"""

from enum import Enum, auto


class LifeState(Enum):
    ACTIVE = auto()
    DORMANT = auto()
    REPLICATING = auto()
    EVOLVING = auto()
    TERMINATED = auto()