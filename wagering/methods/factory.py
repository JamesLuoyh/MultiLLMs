"""
Factory for loading wagering methods from configuration.
"""

import logging
from typing import Dict, Any, Optional


log = logging.getLogger("wagering")


def load_wagering_method(
    method_name: str,
    num_models: int,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Load a wagering method by name.
    
    Args:
        method_name: Name of the wagering method (e.g., "equal_wagers")
        num_models: Number of models in the ensemble
        config: Optional configuration dictionary
        
    Returns:
        WageringMethod instance
        
    Raises:
        ValueError: If method_name is unknown
    """
    config = config or {}
    
    # Import methods locally to avoid circular imports
    from .equal_wagers import EqualWagers
    from .centralized_wagers import CentralizedWagers
    from .decentralized_wagers import DecentralizedWagers
    from .weighted_score_wagers import WeightedScoreWagers
    from .kelly_wagers import KellyWagers
    from .br_regret_wagers import BrRegretWagers
    from .zero_one_wagers import ZeroOneWagers
    from .one_zero_wagers import OneZeroWagers
    from .moe_wagers import MoEWagers
    from .mse_br_wagers import MSEBrWagers
    from .mse_br_wagers_v2 import MSEBrWagersV2

    
    # Built-in methods mapping
    methods = {
        "equal_wagers": EqualWagers,
        "equal": EqualWagers,
        "zero_one_wagers": ZeroOneWagers,
        "one_zero_wagers": OneZeroWagers,
        "centralized_wagers": CentralizedWagers,
        "decentralized_wagers": DecentralizedWagers,
        "weighted_score_wagers": WeightedScoreWagers,
        "kelly_wagers": KellyWagers,
        "br_regret_wagers": BrRegretWagers,
        "moe_wagers": MoEWagers,
        "mse_br_wagers": MSEBrWagers,
        "mse_br_wagers_v2": MSEBrWagersV2,
    }
    
    if method_name in methods:
        return methods[method_name](num_models=num_models, config=config)
    
    raise ValueError(
        f"Unknown wagering method: {method_name}. "
        f"Available methods: {list(methods.keys())}"
    )

