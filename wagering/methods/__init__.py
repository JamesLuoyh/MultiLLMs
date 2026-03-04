"""
Wagering methods for multi-LLM ensemble learning.

This module provides base classes and implementations for generating weights/wagers
for multiple LLMs based on questions, models, or both.
"""

from .base import WageringMethod
from .factory import load_wagering_method
from .equal_wagers import EqualWagers
from .centralized_wagers import CentralizedWagers
from .decentralized_wagers import DecentralizedWagers
from .moe_wagers import MoEWagers
from .kelly_wagers import KellyWagers
from .br_regret_wagers import BrRegretWagers
from .mse_br_wagers import MSEBrWagers
from .weighted_score_wagers import WeightedScoreWagers
from .one_zero_wagers import OneZeroWagers
from .zero_one_wagers import ZeroOneWagers

__all__ = [
    "WageringMethod",
    "load_wagering_method",
    "EqualWagers",
    "CentralizedWagers",
    "DecentralizedWagers",
    "MoEWagers",
    "KellyWagers",
    "BrRegretWagers",
    "MSEBrWagers",
    "WeightedScoreWagers",
    "OneZeroWagers",
    "ZeroOneWagers",
]