import torch

def compute_scoring_rule(model_probs: torch.Tensor, outcome: int, scoring_rule: str) -> torch.Tensor:
    """
    Compute strictly proper scoring rule s(p, omega) for each model.
    Supports both single samples and batches.
    
    Args:
        model_probs: Tensor of shape [num_models, num_options] (single sample)
                     or [batch_size, num_models, num_options] (batch)
        outcome: Integer index of the true outcome omega (single sample)
                 or Tensor of shape [batch_size] with outcome indices (batch)
        scoring_rule: The scoring rule to apply (e.g., 'logarithmic')
    
    Returns:
        scores: Tensor of shape [num_models] (single sample)
                or [batch_size, num_models] (batch)
    """
    # Detect batch mode
    is_batch = model_probs.ndim == 3  # [batch_size, num_models, num_options]
    
    if is_batch:
        # Batch mode: [batch_size, num_models, num_options]
        batch_size, num_models, num_options = model_probs.shape
        
        if scoring_rule == "logarithmic":
            # Logarithmic scoring rule: s(p, omega) = log(p[omega])
            # outcome should be [batch_size]
            batch_indices = torch.arange(batch_size, device=model_probs.device)
            scores = torch.log(model_probs[batch_indices, :, outcome] + 1e-10)  # [batch_size, num_models]
        elif scoring_rule == "brier":
            # Brier score: mean squared error between predicted probabilities and actual outcome
            outcome_one_hot = (torch.arange(num_options, device=model_probs.device).view(1, 1, -1) == 
                              outcome.view(batch_size, 1, 1)).float()  # [batch_size, 1, num_options]
            scores = 1 - ((model_probs - outcome_one_hot) ** 2).mean(dim=2)  # [batch_size, num_models]
        else:
            raise ValueError(f"Unknown scoring rule: {scoring_rule}")
        
        return scores
    else:
        # Single sample mode: [num_models, num_options]
        if scoring_rule == "logarithmic":
            # Logarithmic scoring rule: s(p, omega) = log(p[omega])
            scores = torch.log(model_probs[:, outcome] + 1e-10)
        elif scoring_rule == "brier":
            # Brier score: mean squared error between predicted probabilities and actual outcome
            scores = 1 -((model_probs - (torch.arange(model_probs.size(1)) == outcome).float().view(1, -1).to(model_probs.device)) ** 2).mean(dim=1)
        else:
            raise ValueError(f"Unknown scoring rule: {scoring_rule}")
        
        return scores
