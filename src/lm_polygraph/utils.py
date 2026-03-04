import torch

def compute_scoring_rule(model_probs: torch.Tensor, outcome: int, scoring_rule: str) -> torch.Tensor:
    """
    Compute strictly proper scoring rule s(p, omega) for each model.
    
    Args:
        model_probs: Tensor of shape [num_models, num_options] with model probabilities
        outcome: Integer index of the true outcome omega
        scoring_rule: The scoring rule to apply (e.g., 'logarithmic')
    
    Returns:
        scores: Tensor of shape [num_models] with scores for each model
    """
    if scoring_rule == "logarithmic":
        # Logarithmic scoring rule: s(p, omega) = log(p[omega])
        scores = torch.log(model_probs[:, outcome] + 1e-10)
    else:
        raise ValueError(f"Unknown scoring rule: {scoring_rule}")
    
    return scores
