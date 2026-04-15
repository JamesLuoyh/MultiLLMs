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
from .mse_br_wagers_v2_augmented import MSEBrWagersV2Augmented
from .mse_br_wagers_v3_augmented import MSEBrWagersV3Augmented
from .pre_inference_mse_br_wagers_v2 import PreInferenceMSEBrWagersV2
from .route_llm_bert import RouteLLMBertWagers
from .router_dc import RouterDCWagers
from .nirt_router import NIRTRouterWagers
from .weighted_score_wagers import WeightedScoreWagers
from .one_zero_wagers import OneZeroWagers
from .zero_one_wagers import ZeroOneWagers
from .kl_uniform_wagers import KLUniformWagers
from .packllm_perplexity_wagers import PackLLMPerplexityWagers

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
    "MSEBrWagersV2Augmented",
    "MSEBrWagersV3Augmented",
    "PreInferenceMSEBrWagersV2",
    "RouteLLMBertWagers",
    "RouterDCWagers",
    "NIRTRouterWagers",
    "WeightedScoreWagers",
    "OneZeroWagers",
    "ZeroOneWagers",
    "KLUniformWagers",
    "PackLLMPerplexityWagers",
]