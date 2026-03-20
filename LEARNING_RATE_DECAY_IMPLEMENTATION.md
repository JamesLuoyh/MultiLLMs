# Learning Rate Decay Implementation

## Overview
Implemented learning rate decay with decay factor as a hyperparameter across the training pipeline. This allows the learning rate to be reduced over time during training, which can improve convergence and final model performance.

## Changes Made

### 1. Calibration Manager (`wagering/calibration/manager.py`)
- **Added hyperparameters**:
  - `lr_decay_factor` (default: 1.0) - Factor to multiply learning rate by each decay step
  - `lr_decay_steps` (default: 1) - Number of optimizer steps between decay applications
  
- **Implementation**:
  - Created `torch.optim.lr_scheduler.StepLR` after optimizer initialization
  - Called `scheduler.step()` after each `optimizer.step()` in the training loop
  - Works with the temperature scaling head training process

### 2. MSE BR Wagers V3 (`wagering/methods/mse_br_wagers_v3.py`)
- **Added hyperparameters**:
  - `lr_decay_factor` (default: 1.0) - Factor to multiply learning rate by each decay step
  - `lr_decay_steps` (default: 1) - Number of optimizer steps between decay applications

- **Implementation**:
  - Added `self.schedulers` list to store schedulers alongside optimizers
  - Created `torch.optim.lr_scheduler.StepLR` for each per-model optimizer
  - Updated the optimizer step loop to also call `scheduler.step()`
  - Maintains one scheduler per model/optimizer pair

### 3. Example Configuration Files
Updated YAML configuration files to include the new hyperparameters:

- `examples/configs/wagering_training/mse_br_wagers_v2_1000samples.yaml`
- `examples/configs/wagering_training/mse_br_wagers_v3_1000samples.yaml`
- `examples/configs/wagering_training/calibration/adaptive_temperature_1000samples.yaml`

Default values in configs:
- `lr_decay_factor: 0.95` - Multiply learning rate by 0.95 each step
- `lr_decay_steps: 100` - Apply decay every 100 optimizer steps

## Usage

### Configuration Example
```yaml
wagering_method:
  name: mse_br_wagers_v3
  config:
    learning_rate: 5e-5
    lr_decay_factor: 0.95      # Decay to 95% of current LR
    lr_decay_steps: 100        # Every 100 optimizer steps
```

### Behavior Examples
- `lr_decay_factor: 1.0` - No decay (constant learning rate)
- `lr_decay_factor: 0.95, lr_decay_steps: 100` - Multiply LR by 0.95 every 100 steps
- `lr_decay_factor: 0.5, lr_decay_steps: 50` - Multiply LR by 0.5 every 50 steps (aggressive decay)

## Implementation Details

### How StepLR Works
The implementation uses PyTorch's `StepLR` scheduler, which:
1. Multiplies the learning rate by `gamma` (decay_factor) every `step_size` (decay_steps) optimizer steps
2. Is called after each `optimizer.step()`
3. Can be inspected via `optimizer.param_groups[0]['lr']` to check current learning rate

### Backward Compatibility
- Both hyperparameters have defaults that preserve existing behavior (no decay)
- Existing configs that don't specify these parameters will work unchanged
- Can be incrementally adopted in existing training pipelines

## Testing Recommendations
1. Monitor the learning rate during training with TensorBoard or wandb
2. Compare convergence curves with different decay settings
3. Typical values to try:
   - `lr_decay_factor: 0.95-0.99` for conservative decay
   - `lr_decay_factor: 0.5-0.8` for aggressive decay
   - `lr_decay_steps: 50-200` depending on total training steps

## Future Enhancements
- Support for other scheduler types (ExponentialLR, CosineAnnealingLR, etc.)
- Warmup period before decay starts
- Per-parameter group decay (different rates for different components)
