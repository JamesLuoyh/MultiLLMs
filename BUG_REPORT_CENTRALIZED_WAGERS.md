# Bug Report: Centralized Wagers AUC=0.5 Issue

## Summary
Found and fixed critical bugs in `centralized_wagers.py` that were preventing the router from learning to produce discriminative wagers. This explains why centralized wagers had high accuracy (0.8889) but terrible AUC (0.5000), while mse_br_v3 had lower accuracy (0.7778) but perfect AUC (1.0000).

## Root Cause
The probabilities output by centralized wagers were nearly uniform/random with respect to correctness, leading to AUC=0.5 (indistinguishable from random guessing). This was caused by two bugs in the `update()` method:

## Bugs Found and Fixed

### Bug 1: Wrong Variable Used for Batch Metrics (PRIMARY BUG)
**Location:** `wagering/methods/centralized_wagers.py`, lines 222-224

**Issue:** The `update()` method computes aggregated probabilities from logits and wagers using `LinearPooling.aggregate_torch()`, storing them in `batch_aggregated_probs`. However, the batch metrics computation at the end tried to use the passed-in `aggregated_probs` parameter instead of the computed values:

```python
# BUG: This uses the wrong variable (aggregated_probs parameter)
batch_correct = (np.argmax(aggregated_probs, axis=1) == gold_label)
batch_accuracy = float(np.mean(batch_correct))
avg_prob_correct = float(np.mean(aggregated_probs[np.arange(batch_size), gold_label]))
```

**Impact:** This caused several issues:
1. The parameter `aggregated_probs` might be None, causing crashes
2. If passed, it might have wrong dimensions or be stale (from evaluator, not freshly computed)
3. This prevented the batches from being processed correctly during training

**Fix:** Use the freshly computed `batch_aggregated_probs` instead:
```python
# FIXED: Use the computed probabilities
batch_aggregated_probs_np = batch_aggregated_probs.detach().cpu().numpy()
batch_correct = (np.argmax(batch_aggregated_probs_np, axis=1) == gold_label)
batch_accuracy = float(np.mean(batch_correct))
avg_prob_correct = float(np.mean(batch_aggregated_probs_np[np.arange(batch_size), gold_label]))
```

### Bug 2: Incomplete get_trainable_parameters() Method
**Location:** `wagering/methods/centralized_wagers.py`, line 235

**Issue:** The `get_trainable_parameters()` method only returned router parameters but not the dynamically-created projection parameters:

```python
def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
    """Get list of trainable parameters."""
    return list(self.router.parameters())  # MISSING: projection parameters!
```

**Impact:** 
1. Checkpoint detection logic might fail (if only router params are returned, it might incorrectly determine a checkpoint isn't needed)
2. Incomplete parameter list for debugging/monitoring
3. Inconsistent with `mse_br_v3` which correctly includes both routers and projections

**Fix:** Include projection parameters like in `mse_br_v3`:
```python
def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
    """Get list of trainable parameters (router + projections)."""
    params = list(self.router.parameters())
    for proj in self.model_projections.values():
        params.extend(proj.parameters())
    return params
```

## Evidence
Created test in `test_centralized_debug.py` that demonstrates the fix:
- **Before fix:** Training would crash with `AxisError: axis 1 is out of bounds for array of dimension 1`
- **After fix:** Router successfully learns discriminative wagers:
  - Epoch 0: Wager variance = 0.001242 (nearly uniform)
  - Epoch 100: Wager variance = 0.245427 (highly discriminative)
  - Final state: Router gives ~99% weight to correct model per example

## Impact on Evaluation Results
The centralized wagers method now produces properly calibrated wagers, which should:
1. Fix the AUC=0.5 issue (wagers/confidences will correlate with correctness)
2. Improve calibration metrics (ECE should decrease)
3. May also improve accuracy if the wagers help better aggregate model predictions

The mse_br_v3 results should not be affected as it doesn't have these bugs.

## Recommended Action
1. Apply both fixes to `wagering/methods/centralized_wagers.py`
2. Re-run the PubMedQA evaluation with the fixed centralized wagers
3. Verify that AUC improves significantly from 0.5000
4. Compare ECE and calibration metrics with mse_br_v3
