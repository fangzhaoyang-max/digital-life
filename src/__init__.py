# Digital Life System
# Author: 方兆阳，15岁，2025，进化在这一刻开始

__version__ = "1.9.0"
__author__ = "方兆阳"

from .core.digital_life import TrueDigitalLife
from .core.environment import DigitalEnvironment
from .core.life_states import LifeState

__all__ = ['TrueDigitalLife', 'DigitalEnvironment', 'LifeState']