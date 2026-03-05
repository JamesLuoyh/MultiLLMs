"""
Training pipeline for multi-LLM wagering methods.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import copy
import numpy as np
import torch
import pandas as pd

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset

# Local wagering imports
from wagering.methods.base import WageringMethod
from wagering.training.analytics import WageringAnalytics
from wagering.aggregation.base import AggregationFunction
from wagering.utils.multi_llm_ensemble import (
    collect_option_logits_and_hidden_states_for_model,
    get_cached_logits_and_hidden_states_for_model,
    set_cached_logits_and_hidden_states_for_model,
)

log = logging.getLogger("wagering")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

from sklearn.metrics import roc_auc_score
from lm_polygraph.ue_metrics.ece import ECE


def compute_dynamic_regret(
    model_logits: np.ndarray,
    aggregated_probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute Dynamic Regret (DRegret): aggregated_nll - best_expert_nll.
    
    Args:
        model_logits: [num_examples, num_models, num_options] - logits from each model
        aggregated_probs: [num_examples, num_options] - aggregated probability distributions
        labels: [num_examples] - true labels
        
    Returns:
        (d_regret, best_expert_ids): Average dynamic regret and best expert id per example
    """
    num_examples = model_logits.shape[0]
    num_models = model_logits.shape[1]
    
    # Convert model logits to probabilities
    max_logits = np.max(model_logits, axis=2, keepdims=True)  # [num_examples, num_models, 1]
    stabilized = model_logits - max_logits
    log_z = max_logits + np.log(np.exp(stabilized).sum(axis=2, keepdims=True))
    model_probs = np.exp(model_logits - log_z)  # [num_examples, num_models, num_options]
    
    # Compute NLL for each model on each example
    model_nlls = np.zeros((num_examples, num_models))
    for i in range(num_examples):
        for j in range(num_models):
            model_nlls[i, j] = -np.log(model_probs[i, j, labels[i]] + 1e-10)
    
    # Find best expert (lowest NLL) for each example
    best_expert_ids = np.argmin(model_nlls, axis=1)  # [num_examples]
    best_expert_nlls = model_nlls[np.arange(num_examples), best_expert_ids]  # [num_examples]
    
    # Compute aggregated NLL
    aggregated_nlls = -np.log(aggregated_probs[np.arange(num_examples), labels] + 1e-10)  # [num_examples]
    
    # Dynamic regret = aggregated_nll - best_expert_nll
    d_regret = np.mean(aggregated_nlls - best_expert_nlls)
    
    return d_regret, best_expert_ids


def compute_meta_metrics(
    wagers: np.ndarray,
    best_expert_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute meta metrics treating wagers as predictions of best expert.
    
    Args:
        wagers: [num_examples, num_models] - wager distributions (probability simplex)
        best_expert_ids: [num_examples] - best expert id for each example
        
    Returns:
        Dictionary with meta_acc, meta_nll, meta_auc
    """
    num_examples = wagers.shape[0]
    num_models = wagers.shape[1]
    
    # Meta accuracy: does argmax(wagers) match best expert?
    predicted_expert = np.argmax(wagers, axis=1)  # [num_examples]
    meta_acc = np.mean(predicted_expert == best_expert_ids)
    
    # Meta NLL: -log(wager[best_expert_id])
    meta_nlls = -np.log(wagers[np.arange(num_examples), best_expert_ids] + 1e-10)
    meta_nll = np.mean(meta_nlls)
    
    # Meta AUC: one-vs-rest AUC for each expert, then average
    # Create one-hot labels
    meta_auc = None
    try:
        # For each model, compute binary AUC (is this the best expert?)
        aucs = []
        for model_idx in range(num_models):
            binary_labels = (best_expert_ids == model_idx).astype(int)
            if len(np.unique(binary_labels)) >= 2:  # Need both classes
                auc = roc_auc_score(binary_labels, wagers[:, model_idx])
                aucs.append(auc)
        
        if len(aucs) > 0:
            meta_auc = np.mean(aucs)
        else:
            meta_auc = np.nan
    except Exception as e:
        log.warning(f"Failed to compute meta_auc: {e}")
        meta_auc = np.nan
    
    return {
        "meta_acc": meta_acc,
        "meta_nll": meta_nll,
        "meta_auc": meta_auc if not np.isnan(meta_auc) else None,
    }


class WageringTrainer:
    """
    Trainer for multi-LLM wagering methods.
    
    Handles training loop, logging, checkpointing, and evaluation.
    """
    
    def __init__(
        self,
        models: List[WhiteboxModel],
        datasets: List[Dataset],
        wagering_method: WageringMethod,
        aggregation_function: AggregationFunction,
        option_tokens: List[str] = ["A", "B", "C", "D"],
        checkpoint_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None,
        save_every: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None,
        shuffle_data: bool = True,
        shuffle_seed: int = 42,
        early_stopping_patience: int = 10,  # Now in epochs, default 10 epochs
        batch_size: int = 100,  # Batch size for training loop
        validation_split_ratio: float = 0.1,  # Fraction of data to use for validation (default: 10%)
    ):
        """
        Initialize the trainer.
        
        Args:
            models: List of WhiteboxModel instances
            datasets: List of Dataset instances (will be concatenated for training)
            wagering_method: WageringMethod instance
            aggregation_function: AggregationFunction instance
            option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
            checkpoint_dir: Directory for saving checkpoints
            wandb_logger: Optional wandb logger
            save_every: Save checkpoint every N batches
        """
        self.models = models
        self.datasets = datasets
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.wandb_logger = wandb_logger
        self.save_every = save_every
        self.metadata = metadata or {}
        self.resume_from_checkpoint = resume_from_checkpoint
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed
        self.early_stopping_patience = early_stopping_patience  # In epochs
        self.batch_size = batch_size
        self.validation_split_ratio = validation_split_ratio
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_step = 0
        self.start_epoch = 0
        self.wagers_history = []
        self.metrics_history = []
        
        # Running average tracker for last 5 batches (for wandb logging)
        self.running_avg_window = 5
        self.batch_metrics_buffer = deque(maxlen=self.running_avg_window)  # Store last N batches of metrics

        # Cache the most recent validation metrics (for final logging fallback)
        self.last_val_metrics: Optional[Dict[str, Any]] = None
        
        # Early stopping state (must be initialized before checkpoint loading)
        # Now epoch-based instead of step-based
        # Note: best_d_regret tracks validation d_regret if validation set exists, otherwise training d_regret
        # d_regret is a loss metric, so lower is better (initialized to infinity)
        self.best_d_regret = float('inf')
        self.best_nash_gap = float('inf')  # For methods that provide Nash gap metric
        self.epochs_since_improvement = 0
        self.early_stopped = False
        self.best_wagering_method_state = None  # Store the best checkpoint state
        self.best_epoch = None  # Track which epoch had the best checkpoint
        
        # Collect per-dataset cached logits/hidden states first, then combine datasets and shuffle
        self._collect_logits()
        self._collect_hidden_states()
        self._prepare_datasets()
        # Sanity check: combined dataset length must match cached logits/hidden states
        if hasattr(self, "all_model_logits") and self.all_model_logits is not None:
            combined_len = len(self.combined_dataset.x)
            if self.all_model_logits.shape[1] != combined_len:
                raise RuntimeError(
                    f"Combined dataset size ({combined_len}) does not match cached logits size "
                    f"({self.all_model_logits.shape[1]})."
                )
        if hasattr(self, "all_hidden_states") and self.all_hidden_states is not None:
            combined_len = len(self.combined_dataset.x)
            if isinstance(self.all_hidden_states, list):
                for i, hs in enumerate(self.all_hidden_states):
                    if hs.shape[0] != combined_len:
                        raise RuntimeError(
                            f"Combined dataset size ({combined_len}) does not match cached hidden states "
                            f"for model {i} ({hs.shape[0]})."
                        )
            else:
                if self.all_hidden_states.shape[1] != combined_len:
                    raise RuntimeError(
                        f"Combined dataset size ({combined_len}) does not match cached hidden states size "
                        f"({self.all_hidden_states.shape[1]})."
                    )
        self._apply_shuffling()

    def _get_wandb_run_step(self) -> Optional[int]:
        """Return current wandb run step if available and parseable."""
        if not self.wandb_logger:
            return None

        if hasattr(self.wandb_logger, 'run') and self.wandb_logger.run is not None:
            run = self.wandb_logger.run
            if hasattr(run, 'step') and run.step is not None:
                try:
                    return int(run.step)
                except (TypeError, ValueError):
                    return None

        return None

    def _advance_wandb_plot_step(self) -> int:
        """Advance plot logging step while staying monotonic with run.step."""
        next_step = self.current_step + 1
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            next_step = max(next_step, run_step + 1)
        self.current_step = next_step
        return self.current_step

    def _log_wandb_plot(self, payload: Dict[str, Any]) -> None:
        """Log plot payload to wandb with a safe monotonically increasing step."""
        if not self.wandb_logger:
            return

        plot_step = self._advance_wandb_plot_step()
        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
            self.wandb_logger.run.log(payload, step=plot_step, commit=True)
        else:
            self.wandb_logger.log(payload, step=plot_step, commit=True)

        # Keep local step aligned with wandb's internal run.step, which can advance
        # by one after commit=True logs.
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            self.current_step = max(self.current_step, run_step)

    def _resolve_training_dataset_names(self) -> Tuple[List[str], List[str]]:
        """Return display names and slugified keys for training datasets.
        
        Returns:
            (display_names, slug_names): two lists aligned with self.datasets
        """
        display_names: List[str] = []
        if isinstance(self.metadata, dict):
            for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                v = self.metadata.get(key)
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    display_names = [str(x) for x in v][: len(self.datasets)]
                    break
                if isinstance(v, str) and len(self.datasets) == 1:
                    display_names = [v]
                    break
        if not display_names:
            inferred: List[str] = []
            for i, ds in enumerate(self.datasets):
                ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
            display_names = inferred[: len(self.datasets)]
        if len(display_names) != len(self.datasets):
            display_names = [f"dataset_{i}" for i in range(len(self.datasets))]

        # Slugify for wandb keys
        def slugify(name: str, fallback: str) -> str:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
            return slug if slug else fallback

        slug_names: List[str] = [slugify(display_names[i], f"dataset_{i}") for i in range(len(display_names))]
        return display_names, slug_names
        
    def _prepare_datasets(self):
        """Concatenate all datasets WITHOUT shuffling (after per-dataset cache collection).
        
        Shuffling and train/validation split happen AFTER cache loading in _apply_shuffling().
        """
        all_x = []
        all_y = []
        dataset_indices = []  # Track which dataset each example came from
        
        for dataset_idx, dataset in enumerate(self.datasets):
            all_x.extend(dataset.x)
            all_y.extend(dataset.y)
            # Track dataset index for each example
            dataset_indices.extend([dataset_idx] * len(dataset.x))
        
        # Convert labels to indices if needed
        labels = []
        for y in all_y:
            if isinstance(y, str):
                idx = self.option_tokens.index(y)
            else:
                idx = int(y)
            labels.append(idx)
        
        # Store unshuffled data (will be used for cache key generation)
        batch_size = self.datasets[0].batch_size if self.datasets else 8
        self.combined_dataset = Dataset(all_x, all_y, batch_size=batch_size)
        self.labels = np.array(labels, dtype=np.int32)
        self.dataset_indices = np.array(dataset_indices, dtype=np.int32)
    def _apply_shuffling(self):
        """Apply shuffling to cached arrays and create train/validation splits.
        
        This is called AFTER cache loading so cache keys are based on unshuffled data.
        Shuffles:
        - Dataset (x, y, labels)
        - all_model_logits (if exists)
        - all_hidden_states (if exists)
        Then creates train/validation splits.
        """
        if not self.shuffle_data:
            # No shuffling requested - just create train/validation splits in original order
            log.debug("Shuffling disabled - using original order")
            indices = np.arange(len(self.combined_dataset.x))
        else:
            # Generate shuffle indices
            rng = np.random.RandomState(self.shuffle_seed)
            indices = np.arange(len(self.combined_dataset.x))
            rng.shuffle(indices)
            log.debug(f"Shuffled dataset with seed {self.shuffle_seed}")
        
        # Shuffle dataset
        shuffled_x = [self.combined_dataset.x[i] for i in indices]
        shuffled_y = [self.combined_dataset.y[i] for i in indices]
        shuffled_labels = self.labels[indices]
        shuffled_dataset_indices = self.dataset_indices[indices]
        
        # Shuffle cached logits if they exist
        if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
            # all_model_logits shape: [num_models, num_examples, num_options]
            # Shuffle along the num_examples dimension (axis=1)
            self.all_model_logits = self.all_model_logits[:, indices, :]
            log.debug("Shuffled cached logits")
        
        # Shuffle cached hidden states if they exist
        if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
            if isinstance(self.all_hidden_states, list):
                # List of arrays: shuffle each array
                self.all_hidden_states = [hs[indices, :] for hs in self.all_hidden_states]
            else:
                # Single array: [num_models, num_examples, hidden_dim] or [num_examples, hidden_dim]
                if self.all_hidden_states.ndim == 3:
                    self.all_hidden_states = self.all_hidden_states[:, indices, :]
                else:
                    self.all_hidden_states = self.all_hidden_states[indices, :]
            log.debug("Shuffled cached hidden states")
        
        # Create train/validation splits AFTER shuffling
        batch_size = self.combined_dataset.batch_size
        total_size = len(shuffled_x)
        
        log.debug(f"Creating train/validation split: validation_split_ratio={self.validation_split_ratio}, total_size={total_size}")
        
        if self.validation_split_ratio > 0 and self.validation_split_ratio < 1:
            val_size = int(total_size * self.validation_split_ratio)
            train_size = total_size - val_size
            
            # Split the shuffled data
            train_x = shuffled_x[:train_size]
            train_y = shuffled_y[:train_size]
            train_labels = shuffled_labels[:train_size]
            train_dataset_indices = shuffled_dataset_indices[:train_size]
            
            val_x = shuffled_x[train_size:]
            val_y = shuffled_y[train_size:]
            val_labels = shuffled_labels[train_size:]
            val_dataset_indices = shuffled_dataset_indices[train_size:]
            
            self.combined_dataset = Dataset(train_x, train_y, batch_size=batch_size)
            self.labels = np.array(train_labels, dtype=np.int32)
            self.dataset_indices = np.array(train_dataset_indices, dtype=np.int32)
            
            self.validation_dataset = Dataset(val_x, val_y, batch_size=batch_size)
            self.validation_labels = np.array(val_labels, dtype=np.int32)
            self.validation_dataset_indices = np.array(val_dataset_indices, dtype=np.int32)
            
            log.debug(f"Created validation_dataset with {len(self.validation_dataset.x)} examples")
            
            # Split cached logits if they exist
            if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
                self.all_model_val_logits = self.all_model_logits[:, train_size:, :]
                self.all_model_logits = self.all_model_logits[:, :train_size, :]
                log.debug(f"Split logits: training={self.all_model_logits.shape}, validation={self.all_model_val_logits.shape if self.all_model_val_logits is not None else 'None'}")
            else:
                raise Exception("No all_model_logits to split for validation")
            
            # Split cached hidden states if they exist
            if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                if isinstance(self.all_hidden_states, list):
                    self.all_val_hidden_states = [hs[train_size:, :] for hs in self.all_hidden_states]
                    self.all_hidden_states = [hs[:train_size, :] for hs in self.all_hidden_states]
                else:
                    if self.all_hidden_states.ndim == 3:
                        self.all_val_hidden_states = self.all_hidden_states[:, train_size:, :]
                        self.all_hidden_states = self.all_hidden_states[:, :train_size, :]
                    else:
                        self.all_val_hidden_states = self.all_hidden_states[train_size:, :]
                        self.all_hidden_states = self.all_hidden_states[:train_size, :]
            
            log.debug(f"Split dataset after shuffling: {train_size} train, {val_size} validation ({self.validation_split_ratio*100:.1f}% validation)")
        else:
            # No validation split - use all shuffled data for training
            self.combined_dataset = Dataset(shuffled_x, shuffled_y, batch_size=batch_size)
            self.labels = shuffled_labels
            self.dataset_indices = shuffled_dataset_indices
            self.validation_dataset = None
            self.validation_labels = None
            self.validation_dataset_indices = None
            
            # No need to split cached arrays - they're already shuffled
            if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
                self.all_model_val_logits = None
            if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                self.all_val_hidden_states = None
            
            log.debug(f"Shuffled dataset: {len(self.combined_dataset.x)} examples (no validation split)")
    
    def _compute_grouped_metrics(
        self,
        predictions: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        dataset_indices: np.ndarray,
        wagers_history: Optional[np.ndarray] = None,
        model_logits: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics grouped by original dataset.
        
        Args:
            predictions: Array of predictions
            probs: Array of probability distributions
            labels: Array of true labels
            dataset_indices: Array indicating which dataset each example came from
            wagers_history: Optional array of wagers [num_examples, num_models]
            model_logits: Optional array of model logits [num_examples, num_models, num_options] for computing d_regret
            
        Returns:
            Dictionary mapping dataset index to metrics dictionary
        """
        grouped_metrics = {}
        
        for dataset_idx in range(len(self.datasets)):
            # Get mask for this dataset
            mask = dataset_indices == dataset_idx
            if not np.any(mask):
                continue
            
            # Extract data for this dataset
            dataset_preds = predictions[mask]
            dataset_probs = probs[mask]
            dataset_labels = labels[mask]
            
            # Compute accuracy
            accuracy = np.mean(dataset_preds == dataset_labels)
            
            # Compute NLL
            correct_class_probs = dataset_probs[np.arange(len(dataset_labels)), dataset_labels]
            nll = -np.mean(np.log(correct_class_probs + 1e-10))
            
            # Compute ECE
            ece = None
            try:
                ece_metric = ECE(normalize=True, n_bins=20)
                confidences = dataset_probs.max(axis=1)
                correctness = (dataset_preds == dataset_labels).astype(float)
                ece = ece_metric(confidences.tolist(), correctness.tolist())
            except Exception as e:
                ece = np.nan
            
            # Compute AUC
            auc = None
            max_probs = dataset_probs.max(axis=1)
            correctness = (dataset_preds == dataset_labels).astype(int)
            if len(np.unique(correctness)) >= 2:
                try:
                    auc = roc_auc_score(correctness, max_probs)
                except ValueError:
                    auc = np.nan
            else:
                auc = np.nan
            
            # Compute Dynamic Regret and Meta Metrics if model_logits and wagers provided
            d_regret = None
            meta_acc = None
            meta_nll = None
            meta_auc = None
            if model_logits is not None and wagers_history is not None:
                try:
                    dataset_model_logits = model_logits[mask]
                    dataset_wagers = wagers_history[mask]
                    d_regret, best_expert_ids = compute_dynamic_regret(
                        dataset_model_logits, dataset_probs, dataset_labels
                    )
                    meta_metrics = compute_meta_metrics(dataset_wagers, best_expert_ids)
                    meta_acc = meta_metrics["meta_acc"]
                    meta_nll = meta_metrics["meta_nll"]
                    meta_auc = meta_metrics["meta_auc"]
                except Exception as e:
                    log.warning(f"Could not compute d_regret/meta metrics for dataset {dataset_idx}: {e}")
            
            grouped_metrics[dataset_idx] = {
                "accuracy": float(accuracy),
                "nll": float(nll),
                "ece": float(ece) if not np.isnan(ece) else None,
                "auc": float(auc) if not np.isnan(auc) else None,
                "d_regret": float(d_regret) if d_regret is not None and not np.isnan(d_regret) else None,
                "meta_acc": float(meta_acc) if meta_acc is not None and not np.isnan(meta_acc) else None,
                "meta_nll": float(meta_nll) if meta_nll is not None and not np.isnan(meta_nll) else None,
                "meta_auc": float(meta_auc) if meta_auc is not None and not np.isnan(meta_auc) else None,
                "num_examples": int(np.sum(mask)),
            }
            
            # Add average wagers if provided
            if wagers_history is not None:
                dataset_wagers = wagers_history[mask]
                for model_idx in range(dataset_wagers.shape[1]):
                    grouped_metrics[dataset_idx][f"avg_wager_model_{model_idx}"] = float(np.mean(dataset_wagers[:, model_idx]))
        
        return grouped_metrics
    
    def _compute_running_averages(self) -> Dict[str, float]:
        """
        Compute running averages over the last N batches stored in buffer.
        
        Returns:
            Dictionary with running average metrics
        """
        if len(self.batch_metrics_buffer) == 0:
            return {}
        
        # Collect all metric keys from all batches
        all_keys = set()
        for batch_metrics in self.batch_metrics_buffer:
            all_keys.update(batch_metrics.keys())
        
        # Compute averages for each metric
        running_avgs = {}
        for key in all_keys:
            values = []
            for batch_metrics in self.batch_metrics_buffer:
                if key in batch_metrics:
                    values.append(batch_metrics[key])
            
            if len(values) > 0:
                running_avgs[key] = float(np.mean(values))
        
        return running_avgs
    
    def _plot_validation_wagers_by_dataset(
        self,
        val_wagers: np.ndarray,
        val_results: Dict[str, Any],
    ):
        """
        Plot average validation wagers grouped by dataset.
        
        Args:
            val_wagers: np.ndarray of shape [num_val_examples, num_models] with wager values
            val_results: Dictionary containing 'dataset_indices' for grouping
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if "dataset_indices" not in val_results or self.checkpoint_dir is None:
            return
        
        dataset_indices = val_results["dataset_indices"]
        num_datasets = len(self.datasets)
        num_models = val_wagers.shape[1]
        
        # Get model names
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Get dataset names
        def _resolve_validation_dataset_names() -> List[str]:
            names: List[str] = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names
        
        dataset_names = _resolve_validation_dataset_names()
        
        # Plot: Average wagers per dataset (bar plot)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                mask = dataset_indices == dataset_idx
                if np.any(mask):
                    avg_wager = np.mean(val_wagers[mask, i])
                else:
                    avg_wager = 0.0
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers by Dataset (Validation)", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if self.checkpoint_dir:
            avg_save_path = self.checkpoint_dir / "average_wagers_by_dataset_validation.png"
            plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved average validation wagers by dataset plot to {avg_save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/average_by_dataset/val": wandb.Image(str(avg_save_path))})
        
        plt.close()
    
    def _evaluate_validation(self) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Evaluate the wagering method on the validation set using batch processing.
        
        Returns:
            Tuple of:
                - metrics dictionary (accuracy, nll, ece, auc, ...)
                - val_nash_gaps averaged across validation examples per model (or None)
                - val_score_diffs per validation sample and model (or None)
                - val_wagers per validation sample and model
        """
        # Debug: Check validation state
        has_val_dataset = self.validation_dataset is not None
        has_val_logits = hasattr(self, 'all_model_val_logits') and self.all_model_val_logits is not None
        
        log.debug(f"_evaluate_validation state: validation_dataset={has_val_dataset}, all_model_val_logits={has_val_logits}")
        
        if not has_val_dataset:
            raise RuntimeError("No validation_dataset set - cannot evaluate validation metrics")
        
        if not has_val_logits:
            raise RuntimeError("No all_model_val_logits set - cannot evaluate validation metrics. This may happen if no validation split was configured.")
        
        log.info("Evaluating on validation set...")
        
        # Set wagering method to eval mode (no gradient updates)
        self.wagering_method.eval_mode()
        
        val_predictions = []
        val_probs = []
        val_wagers = []  # Track wagers for each example
        num_val_examples = len(self.validation_dataset.x)
        val_nash_gaps = np.zeros((num_val_examples, len(self.models)))  # Track Nash gaps if provided by wagering method
        val_score_diffs = np.zeros((num_val_examples, len(self.models)))  # Track score differences if provided by wagering method
        eval_batch_size = self.batch_size  # Process validation in batches
        
        for batch_start in range(0, num_val_examples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, num_val_examples)
            batch_size_actual = batch_end - batch_start
            
            # Get batch of logits
            batch_logits = self.all_model_val_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
            batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
            batch_labels = self.validation_labels[batch_start:batch_end]  # [batch_size]
            
            # Get questions for batch (for wagering methods that need them)
            batch_questions = self.validation_dataset.x[batch_start:batch_end]  # List of question strings
            
            # Get hidden states for batch if available
            batch_hidden_states = None
            if hasattr(self, 'all_val_hidden_states') and self.all_val_hidden_states is not None:
                if isinstance(self.all_val_hidden_states, list):
                    batch_hidden_states = []
                    for i in range(len(self.all_val_hidden_states)):
                        model_hs = self.all_val_hidden_states[i][batch_start:batch_end, :]
                        batch_hidden_states.append(model_hs)
                else:
                    batch_hidden_states_array = self.all_val_hidden_states[:, batch_start:batch_end, :]
                    # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                    batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
            
            # Compute wagers for batch
                # Variable hidden dimensions per model - use batch heterogeneous processing
            res_dict = self.wagering_method.compute_wagers(
                model_logits=batch_logits_transposed,
                gold_label=batch_labels,
                hidden_states_list=batch_hidden_states,
                questions=batch_questions,
            )  # [batch_size, num_models]
            batch_wagers = res_dict["wagers"]  # [batch_size, num_models]
            batch_nash_gap = res_dict.get("nash_gap", None)
            batch_score_diff = res_dict.get("score_diff", None)
            # Aggregate predictions for batch
            batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                batch_logits_transposed, batch_wagers
            )  # [batch_size, num_options] each
            
            batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
            if batch_nash_gap is not None and val_nash_gaps is not None:
                val_nash_gaps[batch_start:batch_end] = batch_nash_gap
            else:
                val_nash_gaps = None
            if batch_score_diff is not None and val_score_diffs is not None:
                val_score_diffs[batch_start:batch_end] = batch_score_diff
            else:
                val_score_diffs = None
            val_predictions.extend(batch_predictions.tolist())
            val_probs.extend(batch_aggregated_probs.tolist())
            val_wagers.extend(batch_wagers.tolist())
        
        # Convert to arrays
        val_predictions = np.array(val_predictions, dtype=np.int32)
        val_probs = np.stack(val_probs, axis=0)
        val_wagers = np.stack(val_wagers, axis=0)  # [num_val_examples, num_models]
        if val_nash_gaps is not None:
            val_nash_gaps = np.mean(val_nash_gaps, axis=0)  # Average Nash gap per model over validation set
        if val_score_diffs is not None:
            val_score_diffs = np.asarray(val_score_diffs, dtype=np.float32)
        # Compute metrics
        val_accuracy = np.mean(val_predictions == self.validation_labels)
        
        # Compute NLL
        correct_class_probs = val_probs[np.arange(len(self.validation_labels)), self.validation_labels]
        val_nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # Compute ECE
        val_ece = None
        try:
            ece_metric = ECE(normalize=True, n_bins=20)
            confidences = val_probs.max(axis=1)
            correctness = (val_predictions == self.validation_labels).astype(float)
            val_ece = ece_metric(confidences.tolist(), correctness.tolist())
        except Exception as e:
            log.warning(f"Could not compute validation ECE: {e}")
            val_ece = np.nan
        
        # Compute AUC
        val_auc = None
        max_probs = val_probs.max(axis=1)
        correctness = (val_predictions == self.validation_labels).astype(int)
        if len(np.unique(correctness)) >= 2:
            try:
                val_auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute validation AUC (all predictions same class)")
                val_auc = np.nan
        else:
            val_auc = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        val_d_regret = None
        val_meta_acc = None
        val_meta_nll = None
        val_meta_auc = None
        try:
            # Get validation model logits in the right format [num_examples, num_models, num_options]
            val_model_logits = np.transpose(self.all_model_val_logits, (1, 0, 2))
            val_d_regret, best_expert_ids = compute_dynamic_regret(
                val_model_logits, val_probs, self.validation_labels
            )
            meta_metrics = compute_meta_metrics(val_wagers, best_expert_ids)
            val_meta_acc = meta_metrics["meta_acc"]
            val_meta_nll = meta_metrics["meta_nll"]
            val_meta_auc = meta_metrics["meta_auc"]
        except Exception as e:
            log.warning(f"Could not compute validation d_regret/meta metrics: {e}")
        
        # Set back to train mode
        self.wagering_method.train_mode()
        
        metrics = {
            "accuracy": val_accuracy,
            "nll": val_nll,
            "ece": val_ece if val_ece is not None and not np.isnan(val_ece) else None,
            "auc": val_auc if val_auc is not None and not np.isnan(val_auc) else None,
            "d_regret": val_d_regret if val_d_regret is not None and not np.isnan(val_d_regret) else None,
            "meta_acc": val_meta_acc if val_meta_acc is not None and not np.isnan(val_meta_acc) else None,
            "meta_nll": val_meta_nll if val_meta_nll is not None and not np.isnan(val_meta_nll) else None,
            "meta_auc": val_meta_auc if val_meta_auc is not None and not np.isnan(val_meta_auc) else None,
        }
        
        ece_str = f"{val_ece:.4f}" if val_ece is not None and not np.isnan(val_ece) else 'N/A'
        auc_str = f"{val_auc:.4f}" if val_auc is not None and not np.isnan(val_auc) else 'N/A'
        d_regret_str = f"{val_d_regret:.4f}" if val_d_regret is not None and not np.isnan(val_d_regret) else 'N/A'
        meta_acc_str = f"{val_meta_acc:.4f}" if val_meta_acc is not None and not np.isnan(val_meta_acc) else 'N/A'
        log.info(f"Validation metrics: accuracy={val_accuracy:.4f}, nll={val_nll:.4f}, ece={ece_str}, auc={auc_str}, "
                 f"d_regret={d_regret_str}, meta_acc={meta_acc_str}")
        
        # Compute grouped metrics by dataset
        if hasattr(self, 'validation_dataset_indices') and self.validation_dataset_indices is not None:
            # Transpose validation logits to [num_examples, num_models, num_options]
            val_model_logits_transposed = np.transpose(self.all_model_val_logits, (1, 0, 2)) if self.all_model_val_logits is not None else None
            grouped_metrics = self._compute_grouped_metrics(
                val_predictions, val_probs, self.validation_labels, self.validation_dataset_indices, val_wagers, val_model_logits_transposed
            )
            metrics["grouped"] = grouped_metrics
            
            # Log grouped metrics
            for dataset_idx, dataset_metrics in grouped_metrics.items():
                dataset_name = f"dataset_{dataset_idx}"
                log.info(f"Validation metrics for {dataset_name}: accuracy={dataset_metrics['accuracy']:.4f}, "
                        f"nll={dataset_metrics['nll']:.4f}, num_examples={dataset_metrics['num_examples']}")
            
            # Plot validation wagers by dataset
            val_results = {
                "dataset_indices": self.validation_dataset_indices,
            }
            self._plot_validation_wagers_by_dataset(val_wagers, val_results)
        
        return metrics, val_nash_gaps, val_score_diffs, val_wagers
    
    def _collect_logits(self):
        """
        Collect logits AND hidden states from all models per dataset (no combined dataset cache).
        
        Uses the combined function to collect both logits and hidden states in a single forward pass,
        reducing forward passes from 2 to 1 per model.
        
        Uses shared cache to avoid recomputing logits and hidden states for the same models and datasets
        across different wagering methods. This is the default behavior since LLMs are not updated.
        
        Models are assigned to different GPUs (cuda:0, cuda:1, etc.) for parallel execution.
        
        Note: Validation split happens AFTER cache loading in _apply_shuffling(), so this
        only collects logits and hidden states for the full unshuffled datasets.
        
        TODO: Methods that update LLMs during training should disable caching.
        """
        log.info("Collecting logits and hidden states from all models (per-model, per-dataset cache, unshuffled)...")
        
        num_models = len(self.models)
        num_datasets = len(self.datasets)
        
        per_dataset_logits = []  # List of [num_models, num_examples_ds, num_options]
        per_dataset_hidden_states = []  # List of List[num_models] each [num_examples_ds, hidden_dim_i]
        
        for dataset_idx, dataset in enumerate(self.datasets):
            log.debug(f"Processing dataset {dataset_idx + 1}/{num_datasets} for cache collection")
            dataset_logits_list = []
            dataset_hidden_states_list = []
            
            for model_idx, model in enumerate(self.models):
                model_path = model if isinstance(model, str) else model.model_path
                cached_logits, cached_hidden_states, cached_labels = get_cached_logits_and_hidden_states_for_model(
                    model_path, dataset, self.option_tokens
                )
                
                if cached_logits is not None and cached_hidden_states is not None:
                    log.debug(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        "Using cached logits and hidden states"
                    )
                    model_logits = cached_logits
                    model_hidden_states = cached_hidden_states
                elif cached_logits is not None:
                    log.debug(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        "Found cached logits but not hidden states - collecting both"
                    )
                    if isinstance(model, str):
                        raise RuntimeError(
                            f"Cache miss for model path {model}. Model must be loaded to collect logits."
                        )
                    model_logits, model_hidden_states, model_labels = collect_option_logits_and_hidden_states_for_model(
                        model, dataset, self.option_tokens
                    )
                    set_cached_logits_and_hidden_states_for_model(
                        model, dataset, self.option_tokens,
                        model_logits, model_hidden_states, model_labels
                    )
                else:
                    if isinstance(model, str):
                        raise RuntimeError(
                            f"Cache miss for model path {model}. Model must be loaded to collect logits."
                        )
                    log.info(
                        f"Model {model_idx + 1}/{num_models}, dataset {dataset_idx + 1}/{num_datasets}: "
                        f"Cache miss - collecting logits and hidden states (device: {model.device()})"
                    )
                    try:
                        model_logits, model_hidden_states, model_labels = collect_option_logits_and_hidden_states_for_model(
                            model, dataset, self.option_tokens
                        )
                        set_cached_logits_and_hidden_states_for_model(
                            model, dataset, self.option_tokens,
                            model_logits, model_hidden_states, model_labels
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error collecting logits and hidden states for model {model_idx + 1} on dataset {dataset_idx + 1}: {e}"
                        ) from e
                
                dataset_logits_list.append(model_logits)
                dataset_hidden_states_list.append(model_hidden_states)
            
            # Stack logits for this dataset: [num_models, num_examples_ds, num_options]
            per_dataset_logits.append(np.stack(dataset_logits_list, axis=0))
            per_dataset_hidden_states.append(dataset_hidden_states_list)
        
        # Combine per-dataset logits along the example dimension
        self.all_model_logits = np.concatenate(per_dataset_logits, axis=1)  # [num_models, num_examples, num_options]
        log.debug(f"All training logits shape (combined): {self.all_model_logits.shape}")
        
        # Combine hidden states per model across datasets
        if num_datasets == 0 or num_models == 0:
            self.all_hidden_states = None
            return
        
        # Validate hidden state dims per model across datasets
        hidden_dims_per_model = [per_dataset_hidden_states[0][m].shape[-1] for m in range(num_models)]
        for dataset_idx in range(1, num_datasets):
            for m in range(num_models):
                dim = per_dataset_hidden_states[dataset_idx][m].shape[-1]
                if dim != hidden_dims_per_model[m]:
                    raise RuntimeError(
                        f"Hidden dimension mismatch for model {m} across datasets: "
                        f"{hidden_dims_per_model[m]} vs {dim} (dataset {dataset_idx})"
                    )
        
        # Concatenate per model
        combined_hidden_states_by_model = []
        for m in range(num_models):
            model_hs = [per_dataset_hidden_states[d][m] for d in range(num_datasets)]
            combined_hidden_states_by_model.append(np.concatenate(model_hs, axis=0))
        
        # Stack if all models share same hidden dimension
        if len(set(hidden_dims_per_model)) == 1:
            self.all_hidden_states = np.stack(combined_hidden_states_by_model, axis=0)
            log.debug(f"All training hidden states shape (combined): {self.all_hidden_states.shape}")
        else:
            log.debug(f"Models have different hidden dimensions: {dict(enumerate(hidden_dims_per_model))}")
            log.debug("Storing hidden states as list (will be handled by wagering method)")
            self.all_hidden_states = combined_hidden_states_by_model
        
        # Note: Validation split happens in _apply_shuffling() after cache loading


    def _collect_hidden_states(self):
        """
        Load hidden states from cache (collected together with logits in _collect_logits).
        
        Since _collect_logits now collects both logits and hidden states together,
        this function just ensures hidden states are loaded from cache if they weren't
        already set during _collect_logits.
        
        Note: Validation split happens AFTER cache loading in _apply_shuffling(), so this
        only loads hidden states for the full unshuffled dataset.
        """
        # Check if wagering method needs hidden states
        if not hasattr(self.wagering_method, 'compute_wagers'):
            return
        
        # If hidden states were already collected in _collect_logits, we're done
        if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
            log.debug("Hidden states already collected in _collect_logits, skipping")
            return
        
        # Hidden states should always be collected with logits in _collect_logits with per-model caching
        raise RuntimeError("Hidden states not found. They should have been collected with logits in _collect_logits. "
                   "Some wagering methods may not work correctly.")
    
    def train(self, num_epochs: int = 100) -> Dict[str, Any]:
        """
        Train the wagering method.
        
        Args:
            num_epochs: Number of epochs to train (default: 100)
            
        Returns:
            Dictionary with training results and metrics
        """
        self.wagering_method.train_mode()
        
        num_examples = len(self.combined_dataset.x)
        num_batches = (num_examples + self.batch_size - 1) // self.batch_size
        
        # Training loop
        batch_metrics = []
        
        # Track epoch-level metrics for early stopping
        epoch_accuracies = []

        # Track validation Nash-gap trajectory and related metrics over epochs
        val_nash_gap_history = []
        val_d_regret_history = []
        val_accuracy_history = []
        val_nash_gap_history_epochs = []

        # Cache validation wagers/score-diff for plotting only one epoch at the end.
        best_val_wagers_for_plot: Optional[np.ndarray] = None
        best_val_score_diff_for_plot: Optional[np.ndarray] = None
        latest_val_wagers_for_plot: Optional[np.ndarray] = None
        latest_val_score_diff_for_plot: Optional[np.ndarray] = None
        latest_val_epoch: Optional[int] = None
        
        # Initialize these lists (will be reset each epoch to only keep final epoch's predictions)
        all_predictions = []
        all_aggregated_probs = []
        wagers_history = []
        
        for epoch in range(self.start_epoch, num_epochs):
            log.debug(f"Epoch {epoch+1}/{num_epochs}")
            
            # Reset predictions/probs/wagers at start of each epoch
            # We only want to keep the final epoch's predictions for evaluation
            all_predictions = []
            all_aggregated_probs = []
            wagers_history = []
            
            # Determine starting batch (for resume from mid-epoch)
            if epoch == self.start_epoch and self.current_step > 0:
                # Resume from the batch we left off at (within this epoch)
                start_batch = (self.current_step % num_examples) // self.batch_size
                start_step = start_batch * self.batch_size
                log.debug(f"Resuming from batch {start_batch} (step {start_step}) in epoch {epoch+1}")
                self._processed_start_idx = start_step
            else:
                start_batch = 0
                start_step = 0
                self._processed_start_idx = 0
            
            # Process epoch in batches
            epoch_predictions = []
            epoch_probs = []
            epoch_correct = 0
            epoch_nll_sum = 0.0
            
            for batch_idx in range(start_batch, num_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, num_examples)
                
                # Process batch
                batch_logits = self.all_model_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
                batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
                batch_labels = self.labels[batch_start:batch_end]  # [batch_size]
                batch_size_actual = batch_end - batch_start
                
                # Get questions for batch (for wagering methods that need them)
                batch_questions = self.combined_dataset.x[batch_start:batch_end]  # List of question strings
                
                # Get hidden states for batch if available
                batch_hidden_states = None
                if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
                    if isinstance(self.all_hidden_states, list):
                        # List of arrays with different dimensions - extract batch for each model
                        # Structure: List of [num_models], where each element is [batch_size, hidden_dim_i]
                        batch_hidden_states = []
                        for i in range(len(self.all_hidden_states)):
                            model_hs = self.all_hidden_states[i][batch_start:batch_end, :]  # [batch_size, hidden_dim_i]
                            batch_hidden_states.append(model_hs)
                        # Keep as list to preserve variable hidden dimensions per model
                        # Will be processed per-model in wagering_method.compute_wagers
                    else:
                        # Stacked array: [num_models, num_examples, hidden_dim]
                        batch_hidden_states_array = self.all_hidden_states[:, batch_start:batch_end, :]  # [num_models, batch_size, hidden_dim]
                        # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                        batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
                
                # Compute wagers for entire batch
                    # Variable hidden dimensions per model - use batch  processing
                res_dict = self.wagering_method.compute_wagers(
                    model_logits=batch_logits_transposed,
                    gold_label=batch_labels,
                    hidden_states_list=batch_hidden_states,
                    questions=batch_questions,
                )  # [batch_size, num_models]
                
                batch_wagers = res_dict["wagers"]
                # Aggregate predictions for entire batch
                batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                    batch_logits_transposed, batch_wagers
                )  # [batch_size, num_options] each
                
                batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
                
                # Update wagering method with batch
                # Convert logits to probabilities for update method
                max_logits = np.max(batch_logits_transposed, axis=2, keepdims=True)  # [batch_size, num_models, 1]
                stabilized = batch_logits_transposed - max_logits
                log_z = max_logits + np.log(np.exp(stabilized).sum(axis=2, keepdims=True))
                batch_model_probs = np.exp(batch_logits_transposed - log_z)  # [batch_size, num_models, num_options]
                
                batch_update_info = self.wagering_method.update(
                    aggregated_probs=batch_aggregated_probs,
                    aggregated_pred=batch_predictions,
                    gold_label=batch_labels,
                    model_probs=batch_model_probs,
                    model_logits=batch_logits_transposed,
                    hidden_states=batch_hidden_states,
                )
                

                # Compute batch metrics using vectorized operations
                batch_correct = (batch_predictions == batch_labels)
                batch_nll = -np.log(batch_aggregated_probs[np.arange(batch_size_actual), batch_labels] + 1e-10)
                
                epoch_correct += int(np.sum(batch_correct))
                epoch_nll_sum += np.sum(batch_nll)
                
                # Store batch results for epoch metrics
                all_predictions.extend(batch_predictions.tolist())
                all_aggregated_probs.extend(batch_aggregated_probs.tolist())
                wagers_history.extend(batch_wagers.tolist())
                epoch_predictions.extend(batch_predictions.tolist())
                epoch_probs.extend(batch_aggregated_probs.tolist())
                
                # Log batch-level metrics to wandb
                if self.wandb_logger:
                    global_step = epoch * num_examples + batch_end
                    wandb_log_dict = {
                        "train/batch/accuracy": float(np.mean(batch_correct)),
                        "train/batch/nll": float(np.mean(batch_nll)),
                        "train/batch/batch_size": batch_size_actual,
                    }
                    
                    # Add average wager statistics
                    for i in range(batch_wagers.shape[1]):
                        wandb_log_dict[f"train/batch/wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                    
                    # Add update info if available
                    if batch_update_info:
                        for key, value in batch_update_info.items():
                            if isinstance(value, (int, float, np.number)):
                                wandb_log_dict[f"train/batch/update_{key}"] = float(value)
                    
                    self.wandb_logger.log(wandb_log_dict, step=global_step)
                    
                    # Add to buffer for running averages
                    self.batch_metrics_buffer.append({
                        "batch_accuracy": float(np.mean(batch_correct)),
                        "batch_nll": float(np.mean(batch_nll)),
                        "batch_size": batch_size_actual,
                    })
                    
                    # Compute and log running averages
                    running_avgs = self._compute_running_averages()
                    wandb_avg_dict = {}
                    for key, value in running_avgs.items():
                        wandb_avg_dict[f"train/batch/running_avg_{key}"] = value
                    self.wandb_logger.log(wandb_avg_dict, step=global_step)
                    
                    # Update current_step to track the latest logged step
                    self.current_step = global_step
                else:
                    # Update current_step even without wandb logger
                    self.current_step = epoch * num_examples + batch_end
            

            # Checkpoint after epoch
            # if self.checkpoint_dir:
            #     self._save_checkpoint(epoch)
                # Clear validation cache after checkpoint to free memory
                
            # Compute epoch-level metrics
            epoch_labels = self.labels[self._processed_start_idx:self._processed_start_idx + len(epoch_predictions)]
            epoch_accuracy = np.mean(np.array(epoch_predictions) == epoch_labels)
            epoch_nll = epoch_nll_sum / len(epoch_predictions)
            
            # Increment current_step to ensure epoch-level logging uses a step after batch logs
            # This prevents wandb warnings about logging to an already-used step
            self.current_step += 1
            
            epoch_accuracies.append(epoch_accuracy)
            log.info(f"Epoch {epoch+1} training accuracy: {epoch_accuracy:.4f}, NLL: {epoch_nll:.4f}")
            
            # Evaluate on validation set if available
            val_metrics = {}
            val_nash_gap = None
            val_nash_gap_max = None
            val_score_diff = None
            val_wagers = None
            if self.validation_dataset is not None:
                val_metrics, val_nash_gap, val_score_diff, val_wagers = self._evaluate_validation()
                val_d_regret = val_metrics.get("d_regret", None)
                if val_metrics:
                    self.last_val_metrics = val_metrics

                if val_wagers is not None and val_score_diff is not None:
                    latest_val_wagers_for_plot = np.asarray(val_wagers, dtype=np.float32).copy()
                    latest_val_score_diff_for_plot = np.asarray(val_score_diff, dtype=np.float32).copy()
                    latest_val_epoch = epoch
            else:
                # Fall back to training d_regret if no validation set
                val_d_regret = epoch_d_regret
                log.info("No validation set available, using training d_regret for early stopping")

            # Check for epsilon Nash equilibrium if wagering method provides Nash gap metric
            if val_nash_gap is not None:
                log.info(f"Validation Nash gap for epoch {epoch+1}: {val_nash_gap}")
                val_nash_gap_max = float(np.max(val_nash_gap))  # Use max Nash gap across models for early stopping
                val_nash_gap_mean = float(np.mean(val_nash_gap))
                val_d_regret_value = val_metrics.get("d_regret", None) if val_metrics else None
                val_accuracy_value = val_metrics.get("accuracy", None) if val_metrics else None
                val_nash_gap_history.append(val_nash_gap_mean)
                val_d_regret_history.append(
                    float(val_d_regret_value)
                    if val_d_regret_value is not None and not np.isnan(val_d_regret_value)
                    else np.nan
                )
                val_accuracy_history.append(
                    float(val_accuracy_value)
                    if val_accuracy_value is not None and not np.isnan(val_accuracy_value)
                    else np.nan
                )
                val_nash_gap_history_epochs.append(epoch + 1)
                
                # if val_nash_gap_max < self.best_nash_gap:
                #     self.best_nash_gap = val_nash_gap_max
                #     self.epochs_since_improvement = 0
                #     # Save the best checkpoint state (in memory and to disk)
                #     # IMPORTANT: Use deep copy to avoid reference issues where subsequent
                #     # training updates would modify the stored checkpoint state
                #     self.best_wagering_method_state = copy.deepcopy(self.wagering_method.state_dict())
                #     self.best_epoch = epoch
                #     if val_wagers is not None and val_score_diff is not None:
                #         best_val_wagers_for_plot = np.asarray(val_wagers, dtype=np.float32).copy()
                #         best_val_score_diff_for_plot = np.asarray(val_score_diff, dtype=np.float32).copy()
        
                #     log.debug(f"Saving best checkpoint state dict keys: {list(self.best_wagering_method_state.keys())}")
        
                #     if self.checkpoint_dir:
                #         best_checkpoint_path = self.checkpoint_dir / "best_checkpoint"
                #         best_checkpoint_path.mkdir(parents=True, exist_ok=True)
                #         self.wagering_method.save_pretrained(str(best_checkpoint_path))
                #         log.debug(f"New best nash gap: {self.best_nash_gap:.4f} at epoch {epoch+1} - saved best checkpoint")
                #     else:
                #         log.debug(f"New best nash gap: {self.best_nash_gap:.4f} at epoch {epoch+1}")
                # else:
                #     self.epochs_since_improvement += 1
                
                # # Check if we should stop early
                # if self.epochs_since_improvement >= self.early_stopping_patience:
                #     log.info(
                #         f"Early stopping: No improvement on validation set for {self.early_stopping_patience} epochs. "
                #         f"Best validation nash gap: {self.best_nash_gap:.4f} (from epoch {self.best_epoch + 1})"
                #     )
                #     self.early_stopped = True
                #     # Load the best checkpoint before breaking
                #     if self.best_wagering_method_state is not None:
                #         log.info(f"Loading best checkpoint from epoch {self.best_epoch + 1} (nash gap={self.best_nash_gap:.4f})")
                #         log.debug(f"State dict keys before load: {list(self.wagering_method.state_dict().keys())}")
                #         self.wagering_method.load_state_dict(self.best_wagering_method_state)
                #         log.debug(f"State dict keys after load: {list(self.wagering_method.state_dict().keys())}")
                #     break
            

            # Early stopping: check for improvement on validation set after each epoch
            # d_regret is a loss metric, so lower is better
            if self.early_stopping_patience > 0 and val_d_regret is not None:
                if val_d_regret < self.best_d_regret:
                    self.best_d_regret = val_d_regret
                    self.epochs_since_improvement = 0
                    # Save the best checkpoint state (in memory and to disk)
                    # IMPORTANT: Use deep copy to avoid reference issues where subsequent
                    # training updates would modify the stored checkpoint state
                    self.best_wagering_method_state = copy.deepcopy(self.wagering_method.state_dict())
                    self.best_epoch = epoch
                    if val_wagers is not None and val_score_diff is not None:
                        best_val_wagers_for_plot = np.asarray(val_wagers, dtype=np.float32).copy()
                        best_val_score_diff_for_plot = np.asarray(val_score_diff, dtype=np.float32).copy()
        
                    log.debug(f"Saving best checkpoint state dict keys: {list(self.best_wagering_method_state.keys())}")
        
                    if self.checkpoint_dir:
                        best_checkpoint_path = self.checkpoint_dir / "best_checkpoint"
                        best_checkpoint_path.mkdir(parents=True, exist_ok=True)
                        self.wagering_method.save_pretrained(str(best_checkpoint_path))
                        log.debug(f"New best d_regret: {self.best_d_regret:.4f} at epoch {epoch+1} - saved best checkpoint")
                    else:
                        log.debug(f"New best d_regret: {self.best_d_regret:.4f} at epoch {epoch+1}")
                else:
                    self.epochs_since_improvement += 1
                
                # Check if we should stop early
                if self.epochs_since_improvement >= self.early_stopping_patience:
                    log.info(
                        f"Early stopping: No improvement on validation set for {self.early_stopping_patience} epochs. "
                        f"Best validation d_regret: {self.best_d_regret:.4f} (from epoch {self.best_epoch + 1})"
                    )
                    self.early_stopped = True
                    # Load the best checkpoint before breaking
                    if self.best_wagering_method_state is not None:
                        log.info(f"Loading best checkpoint from epoch {self.best_epoch + 1} (d_regret={self.best_d_regret:.4f})")
                        log.debug(f"State dict keys before load: {list(self.wagering_method.state_dict().keys())}")
                        self.wagering_method.load_state_dict(self.best_wagering_method_state)
                        log.debug(f"State dict keys after load: {list(self.wagering_method.state_dict().keys())}")
                    break
            
            # Log epoch-level metrics to wandb
            if self.wandb_logger and len(epoch_predictions) > 0:
                epoch_probs_array = np.stack(epoch_probs)
                
                # Compute ECE for epoch
                epoch_ece = None
                try:
                    ece_metric = ECE(normalize=True, n_bins=20)
                    confidences = epoch_probs_array.max(axis=1)
                    correctness = (np.array(epoch_predictions) == epoch_labels).astype(float)
                    epoch_ece = ece_metric(confidences.tolist(), correctness.tolist())
                except Exception as e:
                    epoch_ece = np.nan
                
                # Compute AUC for epoch
                epoch_auc = None
                max_probs = epoch_probs_array.max(axis=1)
                correctness = (np.array(epoch_predictions) == epoch_labels).astype(int)
                if len(np.unique(correctness)) >= 2:
                    try:
                        epoch_auc = roc_auc_score(correctness, max_probs)
                    except ValueError:
                        epoch_auc = np.nan
                else:
                    epoch_auc = np.nan
                
                # Compute Dynamic Regret and Meta Metrics for epoch
                epoch_d_regret = None
                epoch_meta_acc = None
                epoch_meta_nll = None
                epoch_meta_auc = None
                try:
                    # Get epoch model logits in the right format [num_examples, num_models, num_options]
                    epoch_start_idx = self._processed_start_idx
                    epoch_end_idx = epoch_start_idx + len(epoch_predictions)
                    epoch_model_logits_transposed = self.all_model_logits[:, epoch_start_idx:epoch_end_idx, :]  # [num_models, num_examples, num_options]
                    epoch_model_logits = np.transpose(epoch_model_logits_transposed, (1, 0, 2))  # [num_examples, num_models, num_options]
                    epoch_wagers_array = np.array(wagers_history)  # [num_examples, num_models]
                    
                    epoch_d_regret, best_expert_ids = compute_dynamic_regret(
                        epoch_model_logits, epoch_probs_array, epoch_labels
                    )
                    meta_metrics = compute_meta_metrics(epoch_wagers_array, best_expert_ids)
                    epoch_meta_acc = meta_metrics["meta_acc"]
                    epoch_meta_nll = meta_metrics["meta_nll"]
                    epoch_meta_auc = meta_metrics["meta_auc"]
                except Exception as e:
                    log.warning(f"Could not compute epoch d_regret/meta metrics: {e}")
                
                # Log epoch metrics
                wandb_epoch_dict = {
                    "train/epoch/accuracy": epoch_accuracy,
                    "train/epoch/nll": epoch_nll,
                    "train/epoch/ece": epoch_ece if epoch_ece is not None and not np.isnan(epoch_ece) else None,
                    "train/epoch/auc": epoch_auc if epoch_auc is not None and not np.isnan(epoch_auc) else None,
                    "train/epoch/d_regret": epoch_d_regret if epoch_d_regret is not None and not np.isnan(epoch_d_regret) else None,
                    "train/epoch/meta_acc": epoch_meta_acc if epoch_meta_acc is not None and not np.isnan(epoch_meta_acc) else None,
                    "train/epoch/meta_nll": epoch_meta_nll if epoch_meta_nll is not None and not np.isnan(epoch_meta_nll) else None,
                    "train/epoch/meta_auc": epoch_meta_auc if epoch_meta_auc is not None and not np.isnan(epoch_meta_auc) else None,
                    "train/epoch": epoch + 1,
                }
                
                # Add validation metrics if available
                if val_metrics:
                    val_dict_update = {
                        "val/epoch/accuracy": val_metrics.get("accuracy", 0.0),
                        "val/epoch/nll": val_metrics.get("nll", 0.0),
                    }
                    # Only add optional metrics if they're not None/NaN
                    if val_metrics.get("ece") is not None and not np.isnan(val_metrics.get("ece", np.nan)):
                        val_dict_update["val/epoch/ece"] = val_metrics.get("ece")
                    if val_metrics.get("auc") is not None and not np.isnan(val_metrics.get("auc", np.nan)):
                        val_dict_update["val/epoch/auc"] = val_metrics.get("auc")
                    if val_metrics.get("d_regret") is not None and not np.isnan(val_metrics.get("d_regret", np.nan)):
                        val_dict_update["val/epoch/d_regret"] = val_metrics.get("d_regret")
                    if val_metrics.get("meta_acc") is not None and not np.isnan(val_metrics.get("meta_acc", np.nan)):
                        val_dict_update["val/epoch/meta_acc"] = val_metrics.get("meta_acc")
                    if val_metrics.get("meta_nll") is not None and not np.isnan(val_metrics.get("meta_nll", np.nan)):
                        val_dict_update["val/epoch/meta_nll"] = val_metrics.get("meta_nll")
                    if val_metrics.get("meta_auc") is not None and not np.isnan(val_metrics.get("meta_auc", np.nan)):
                        val_dict_update["val/epoch/meta_auc"] = val_metrics.get("meta_auc")
                    if val_nash_gap_max is not None and not np.isnan(val_nash_gap_max):
                        val_dict_update["val/epoch/nash_gap_max"] = val_nash_gap_max
                    
                    wandb_epoch_dict.update(val_dict_update)
                    log.info(f"  Validation accuracy={val_metrics.get('accuracy', 0.0):.4f}, nll={val_metrics.get('nll', 0.0):.4f}")
                    
                    # Add grouped validation metrics if available
                    if "grouped" in val_metrics:
                        _, slug_names = self._resolve_training_dataset_names()
                        grouped_count = 0
                        for dataset_idx, dataset_metrics in val_metrics["grouped"].items():
                            dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                            grouped_update = {
                                f"val/epoch/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                                f"val/epoch/{dataset_key}/nll": dataset_metrics["nll"],
                                f"val/epoch/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                            }
                            # Only add optional metrics if they're not None/NaN
                            if dataset_metrics.get("ece") is not None and not np.isnan(dataset_metrics.get("ece", np.nan)):
                                grouped_update[f"val/epoch/{dataset_key}/ece"] = dataset_metrics.get("ece")
                            if dataset_metrics.get("auc") is not None and not np.isnan(dataset_metrics.get("auc", np.nan)):
                                grouped_update[f"val/epoch/{dataset_key}/auc"] = dataset_metrics.get("auc")
                            wandb_epoch_dict.update(grouped_update)
                            grouped_count += 1
                else:
                    raise RuntimeError(f"Validation was run but val_metrics is empty. validation_dataset={self.validation_dataset is not None}")
                
                # Log to wandb with explicit error handling
                validation_metric_count = sum(1 for k in wandb_epoch_dict if k.startswith('val/'))
                
                try:
                    # Handle both wandb module and wandb.run patterns
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        # wandb module - use wandb.run.log() which is more reliable
                        self.wandb_logger.run.log(wandb_epoch_dict, step=self.current_step)
                    elif hasattr(self.wandb_logger, 'log'):
                        # Either wandb.run object or mock logger
                        self.wandb_logger.log(wandb_epoch_dict, step=self.current_step)
                    else:
                        raise RuntimeError(f"wandb_logger doesn't have 'log' method. Type: {type(self.wandb_logger)}")
                    
                except Exception as e:
                    raise RuntimeError(f"✗ Error logging to wandb: {e}") from e
            else:
                # Fallback: if the main logging block is skipped, still log validation metrics if available
                if self.wandb_logger and val_metrics:
                    wandb_val_dict = {
                        "val/epoch/accuracy": val_metrics.get("accuracy", 0.0),
                        "val/epoch/nll": val_metrics.get("nll", 0.0),
                    }
                    # Only add optional metrics if they're not None/NaN
                    if val_metrics.get("ece") is not None and not np.isnan(val_metrics.get("ece", np.nan)):
                        wandb_val_dict["val/epoch/ece"] = val_metrics.get("ece")
                    if val_metrics.get("auc") is not None and not np.isnan(val_metrics.get("auc", np.nan)):
                        wandb_val_dict["val/epoch/auc"] = val_metrics.get("auc")
                    if val_metrics.get("d_regret") is not None and not np.isnan(val_metrics.get("d_regret", np.nan)):
                        wandb_val_dict["val/epoch/d_regret"] = val_metrics.get("d_regret")
                    if val_metrics.get("meta_acc") is not None and not np.isnan(val_metrics.get("meta_acc", np.nan)):
                        wandb_val_dict["val/epoch/meta_acc"] = val_metrics.get("meta_acc")
                    if val_metrics.get("meta_nll") is not None and not np.isnan(val_metrics.get("meta_nll", np.nan)):
                        wandb_val_dict["val/epoch/meta_nll"] = val_metrics.get("meta_nll")
                    if val_metrics.get("meta_auc") is not None and not np.isnan(val_metrics.get("meta_auc", np.nan)):
                        wandb_val_dict["val/epoch/meta_auc"] = val_metrics.get("meta_auc")
                    if val_nash_gap_max is not None and not np.isnan(val_nash_gap_max):
                        wandb_val_dict["val/epoch/nash_gap_max"] = val_nash_gap_max
                    
                    log.debug(f"Fallback path: Logging {len(wandb_val_dict)} validation metrics to wandb")
                    try:
                        # Handle both wandb module and wandb.run patterns
                        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                            self.wandb_logger.run.log(wandb_val_dict, step=self.current_step)
                        else:
                            self.wandb_logger.log(wandb_val_dict, step=self.current_step)
                        log.debug(f"✓ Successfully logged validation metrics to wandb via fallback path at epoch {epoch + 1}")
                    except Exception as e:
                        raise RuntimeError(f"✗ Error logging validation metrics via fallback: {e}") from e
            
            # Log hyperparameters to wandb config (only once)
            if self.wandb_logger and epoch == 0:
                hyperparams = {}
                
                # Wagering method hyperparameters
                if hasattr(self.wagering_method, 'hidden_dim'):
                    hyperparams["wagering/hidden_dim"] = self.wagering_method.hidden_dim
                if hasattr(self.wagering_method, 'hidden_layers'):
                    hyperparams["wagering/hidden_layers"] = str(self.wagering_method.hidden_layers)
                if hasattr(self.wagering_method, 'learning_rate'):
                    hyperparams["wagering/learning_rate"] = self.wagering_method.learning_rate
                if hasattr(self.wagering_method, 'device_str'):
                    hyperparams["wagering/device"] = self.wagering_method.device_str
                
                # Training hyperparameters
                hyperparams["training/num_models"] = len(self.models)
                hyperparams["training/num_datasets"] = len(self.datasets)
                hyperparams["training/shuffle_data"] = self.shuffle_data
                hyperparams["training/shuffle_seed"] = self.shuffle_seed
                hyperparams["training/early_stopping_patience"] = self.early_stopping_patience
                hyperparams["training/save_every"] = self.save_every
                hyperparams["training/batch_size"] = self.batch_size
                hyperparams["training/validation_split_ratio"] = self.validation_split_ratio
                
                # Aggregation method
                hyperparams["aggregation/name"] = type(self.aggregation_function).__name__
                
                # Log metadata
                if self.metadata:
                    for k, v in self.metadata.items():
                        if k not in ["wagering_config", "aggregation_config"]:
                            hyperparams[f"metadata/{k}"] = v
                
                # Update wandb config
                if hyperparams:
                    self.wandb_logger.config.update(hyperparams)
                    log.debug(f"Logged hyperparameters to wandb: {list(hyperparams.keys())}")
        
        # Ensure best checkpoint is loaded for downstream evaluation/checkpoint saving
        if self.best_wagering_method_state is not None and not self.early_stopped:
            log.debug(
                "Training completed without early stopping. Loading best checkpoint state "
                "for final checkpoint saving and evaluation."
            )
            self.wagering_method.load_state_dict(self.best_wagering_method_state)

        # Plot only one validation wagers-vs-score_diff figure for the best epoch.
        plot_val_wagers = best_val_wagers_for_plot
        plot_val_score_diffs = best_val_score_diff_for_plot
        plot_epoch = self.best_epoch
        if plot_val_wagers is None or plot_val_score_diffs is None or plot_epoch is None:
            plot_val_wagers = latest_val_wagers_for_plot
            plot_val_score_diffs = latest_val_score_diff_for_plot
            plot_epoch = latest_val_epoch
            if plot_val_wagers is not None and plot_val_score_diffs is not None and plot_epoch is not None:
                log.debug(
                    "Best epoch plot data unavailable; using latest validation epoch "
                    f"{plot_epoch + 1} for wagers_vs_score_diff plotting."
                )

        if plot_val_wagers is not None and plot_val_score_diffs is not None and plot_epoch is not None:
            self._plot_val_wagers_vs_score_diff_for_epoch(
                val_wagers=plot_val_wagers,
                val_score_diffs=plot_val_score_diffs,
                epoch=plot_epoch,
            )

        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        
        # Get labels for the examples we actually processed (in case of resume or early stopping)
        # If we resumed, we only processed examples from start_step onwards
        num_processed = len(all_predictions)
        # If we resumed, we need to get the labels for the examples we processed
        # Since we always process sequentially from start_step, we can slice
        if hasattr(self, '_processed_start_idx'):
            start_idx = self._processed_start_idx
            processed_labels = self.labels[start_idx:start_idx + num_processed]
        else:
            # For fresh start or early stopping, we processed examples from 0 to num_processed-1
            processed_labels = self.labels[:num_processed]
        
        # Ensure processed_labels is a numpy array with the correct shape
        processed_labels = np.array(processed_labels, dtype=np.int32)
        
        # Verify shapes match
        if len(all_predictions) != len(processed_labels):
            log.error(
                f"Shape mismatch: all_predictions has {len(all_predictions)} elements, "
                f"but processed_labels has {len(processed_labels)} elements. "
                f"Total dataset size: {len(self.labels)}"
            )
            raise ValueError(
                f"Shape mismatch: predictions ({len(all_predictions)}) vs labels ({len(processed_labels)})"
            )
        
        # Compute final metrics
        accuracy = np.mean(all_predictions == processed_labels)
        
        # Compute NLL (negative log likelihood) for correct classes
        correct_class_probs = all_aggregated_probs[np.arange(len(processed_labels)), processed_labels]
        nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # Compute ECE
        ece = None
        try:
            ece_metric = ECE(normalize=True, n_bins=20)
            confidences = all_aggregated_probs.max(axis=1)
            correctness = (all_predictions == processed_labels).astype(float)
            ece = ece_metric(confidences.tolist(), correctness.tolist())
        except Exception as e:
            log.warning(f"Could not compute ECE: {e}")
            ece = np.nan
        
        # Compute AUC
        auc = None
        max_probs = all_aggregated_probs.max(axis=1)
        correctness = (all_predictions == processed_labels).astype(int)
        if len(np.unique(correctness)) >= 2:
            try:
                auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute AUC (all predictions same class)")
                auc = np.nan
        else:
            auc = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        d_regret = None
        meta_acc = None
        meta_nll = None
        meta_auc = None
        try:
            # Get model logits for processed examples in the right format [num_examples, num_models, num_options]
            if hasattr(self, '_processed_start_idx'):
                start_idx = self._processed_start_idx
                final_model_logits_transposed = self.all_model_logits[:, start_idx:start_idx + num_processed, :]
            else:
                final_model_logits_transposed = self.all_model_logits[:, :num_processed, :]
            
            final_model_logits = np.transpose(final_model_logits_transposed, (1, 0, 2))  # [num_examples, num_models, num_options]
            
            d_regret, best_expert_ids = compute_dynamic_regret(
                final_model_logits, all_aggregated_probs, processed_labels
            )
            meta_metrics = compute_meta_metrics(wagers_history, best_expert_ids)
            meta_acc = meta_metrics["meta_acc"]
            meta_nll = meta_metrics["meta_nll"]
            meta_auc = meta_metrics["meta_auc"]
        except Exception as e:
            log.warning(f"Could not compute final d_regret/meta metrics: {e}")
        
        # Get dataset indices for processed examples
        if hasattr(self, '_processed_start_idx'):
            start_idx = self._processed_start_idx
            processed_dataset_indices = self.dataset_indices[start_idx:start_idx + num_processed]
        else:
            processed_dataset_indices = self.dataset_indices[:num_processed]
        
        # Compute grouped metrics by dataset
        grouped_metrics = self._compute_grouped_metrics(
            all_predictions, all_aggregated_probs, processed_labels,
            processed_dataset_indices, wagers_history, final_model_logits
        )
        
        results = {
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": processed_labels,  # Use processed labels, not all labels
            "dataset_indices": processed_dataset_indices,
            "wagers_history": wagers_history,
            "val_nash_gap_history": np.array(val_nash_gap_history, dtype=np.float32),
            "val_d_regret_history": np.array(val_d_regret_history, dtype=np.float32),
            "val_accuracy_history": np.array(val_accuracy_history, dtype=np.float32),
            "val_nash_gap_history_epochs": np.array(val_nash_gap_history_epochs, dtype=np.int32),
            "batch_metrics": batch_metrics,
            "final_accuracy": accuracy,
            "final_nll": nll,
            "final_ece": ece,
            "final_auc": auc,
            "final_d_regret": d_regret,
            "final_meta_acc": meta_acc,
            "final_meta_nll": meta_nll,
            "final_meta_auc": meta_auc,
            "grouped_metrics": grouped_metrics,
        }
        
        # Log grouped metrics
        display_names, slug_names = self._resolve_training_dataset_names()
        log.info("\n=== Training Metrics by Dataset ===")
        for dataset_idx, dataset_metrics in grouped_metrics.items():
            display_name = display_names[dataset_idx] if dataset_idx < len(display_names) else f"dataset_{dataset_idx}"
            log.info(f"{display_name}: accuracy={dataset_metrics['accuracy']:.4f}, "
                f"nll={dataset_metrics['nll']:.4f}, num_examples={dataset_metrics['num_examples']}")
        
        # Create analytics dataframe
        dataset_size = len(self.combined_dataset.x) if hasattr(self, 'combined_dataset') and self.combined_dataset is not None else None
        analytics_df = WageringAnalytics.create_training_analytics(
            wagering_method=self.wagering_method,
            aggregation_function=self.aggregation_function,
            models=self.models,
            datasets=self.datasets,
            shuffle_data=self.shuffle_data,
            shuffle_seed=self.shuffle_seed,
            early_stopping_patience=self.early_stopping_patience,
            save_every=self.save_every,
            results=results,
            metadata=self.metadata,
            checkpoint_dir=self.checkpoint_dir,
            dataset_size=dataset_size,
        )
        results["analytics_df"] = analytics_df
        
        # Save analytics dataframe to checkpoint directory
        if self.checkpoint_dir:
            analytics_path = self.checkpoint_dir / "analytics.csv"
            analytics_df.to_csv(analytics_path, index=False)
            log.debug(f"Saved analytics dataframe to {analytics_path}")
        
        # Log final training metrics to wandb
        if self.wandb_logger:
            proposed_final_step = self.current_step + 1 if hasattr(self, 'current_step') else num_epochs * num_examples
            wandb_run_step = None
            if (
                hasattr(self.wandb_logger, 'run')
                and self.wandb_logger.run is not None
                and hasattr(self.wandb_logger.run, 'step')
            ):
                try:
                    run_step_value = self.wandb_logger.run.step
                    if run_step_value is not None:
                        wandb_run_step = int(run_step_value)
                except (TypeError, ValueError):
                    wandb_run_step = None

            final_step = (
                max(proposed_final_step, wandb_run_step + 1)
                if wandb_run_step is not None
                else proposed_final_step
            )
            wandb_final_dict = {
                "train/final/accuracy": accuracy,
                "train/final/nll": nll,
                "train/final/ece": ece if ece is not None and not np.isnan(ece) else None,
                "train/final/auc": auc if auc is not None and not np.isnan(auc) else None,
                "train/final/d_regret": d_regret if d_regret is not None and not np.isnan(d_regret) else None,
                "train/final/meta_acc": meta_acc if meta_acc is not None and not np.isnan(meta_acc) else None,
                "train/final/meta_nll": meta_nll if meta_nll is not None and not np.isnan(meta_nll) else None,
                "train/final/meta_auc": meta_auc if meta_auc is not None and not np.isnan(meta_auc) else None,
            }
            
            # Add grouped metrics
            _, slug_names = self._resolve_training_dataset_names()
            for dataset_idx, dataset_metrics in grouped_metrics.items():
                dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                wandb_final_dict.update({
                    f"train/final/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                    f"train/final/{dataset_key}/nll": dataset_metrics["nll"],
                    f"train/final/{dataset_key}/ece": dataset_metrics.get("ece"),
                    f"train/final/{dataset_key}/auc": dataset_metrics.get("auc"),
                    f"train/final/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                })
                
                for model_idx in range(len(self.models)):
                    wager_key = f"avg_wager_model_{model_idx}"
                    if wager_key in dataset_metrics:
                        wandb_final_dict[f"train/final/{dataset_key}/{wager_key}"] = dataset_metrics[wager_key]
            
            try:
                final_plot_step = final_step + 1
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(wandb_final_dict, step=final_step, commit=True)
                    self.wandb_logger.run.log(wandb_final_dict, step=final_plot_step, commit=True)
                else:
                    self.wandb_logger.log(wandb_final_dict, step=final_step, commit=True)
                    self.wandb_logger.log(wandb_final_dict, step=final_plot_step, commit=True)
                self.current_step = max(self.current_step, final_plot_step)
            except Exception as e:
                raise RuntimeError(f"Error logging train/final metrics to wandb: {e}") from e
        
        # Log final validation metrics to wandb
        if self.wandb_logger:
            final_val_metrics = {}
            final_val_nash_gap = None
            if self.validation_dataset is not None:
                final_val_metrics, final_val_nash_gap, _, _ = self._evaluate_validation()

            # Fallback to last available validation metrics if current evaluation is unavailable
            if not final_val_metrics and self.last_val_metrics:
                final_val_metrics = self.last_val_metrics

            if not final_val_metrics:
                raise Exception("Final validation metrics missing; skipping val/final logging.")
            else:
                proposed_final_step = self.current_step + 1
                wandb_run_step = None
                if (
                    hasattr(self.wandb_logger, 'run')
                    and self.wandb_logger.run is not None
                    and hasattr(self.wandb_logger.run, 'step')
                ):
                    try:
                        run_step_value = self.wandb_logger.run.step
                        if run_step_value is not None:
                            wandb_run_step = int(run_step_value)
                    except (TypeError, ValueError):
                        wandb_run_step = None

                final_step = (
                    max(proposed_final_step, wandb_run_step + 1)
                    if wandb_run_step is not None
                    else proposed_final_step
                )
                
                wandb_val_final_dict = {
                    "val/final/accuracy": final_val_metrics.get("accuracy", 0.0),
                    "val/final/nll": final_val_metrics.get("nll", 0.0),
                }
                if final_val_metrics.get("ece") is not None and not np.isnan(final_val_metrics.get("ece", np.nan)):
                    wandb_val_final_dict["val/final/ece"] = final_val_metrics.get("ece")
                if final_val_metrics.get("auc") is not None and not np.isnan(final_val_metrics.get("auc", np.nan)):
                    wandb_val_final_dict["val/final/auc"] = final_val_metrics.get("auc")
                if final_val_metrics.get("d_regret") is not None and not np.isnan(final_val_metrics.get("d_regret", np.nan)):
                    wandb_val_final_dict["val/final/d_regret"] = final_val_metrics.get("d_regret")
                if final_val_metrics.get("meta_acc") is not None and not np.isnan(final_val_metrics.get("meta_acc", np.nan)):
                    wandb_val_final_dict["val/final/meta_acc"] = final_val_metrics.get("meta_acc")
                if final_val_metrics.get("meta_nll") is not None and not np.isnan(final_val_metrics.get("meta_nll", np.nan)):
                    wandb_val_final_dict["val/final/meta_nll"] = final_val_metrics.get("meta_nll")
                if final_val_metrics.get("meta_auc") is not None and not np.isnan(final_val_metrics.get("meta_auc", np.nan)):
                    wandb_val_final_dict["val/final/meta_auc"] = final_val_metrics.get("meta_auc")
                if final_val_nash_gap is not None:
                    final_val_nash_gap_max = float(np.max(final_val_nash_gap))
                    if not np.isnan(final_val_nash_gap_max):
                        wandb_val_final_dict["val/final/nash_gap_max"] = final_val_nash_gap_max
                
                if "grouped" in final_val_metrics:
                    _, slug_names = self._resolve_training_dataset_names()
                    for dataset_idx, dataset_metrics in final_val_metrics["grouped"].items():
                        dataset_key = slug_names[dataset_idx] if dataset_idx < len(slug_names) else f"dataset_{dataset_idx}"
                        wandb_val_final_dict.update({
                            f"val/final/{dataset_key}/accuracy": dataset_metrics["accuracy"],
                            f"val/final/{dataset_key}/nll": dataset_metrics["nll"],
                            f"val/final/{dataset_key}/num_examples": dataset_metrics["num_examples"],
                        })
                        if dataset_metrics.get("ece") is not None and not np.isnan(dataset_metrics.get("ece", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/ece"] = dataset_metrics.get("ece")
                        if dataset_metrics.get("auc") is not None and not np.isnan(dataset_metrics.get("auc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/auc"] = dataset_metrics.get("auc")
                        if dataset_metrics.get("d_regret") is not None and not np.isnan(dataset_metrics.get("d_regret", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/d_regret"] = dataset_metrics.get("d_regret")
                        if dataset_metrics.get("meta_acc") is not None and not np.isnan(dataset_metrics.get("meta_acc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_acc"] = dataset_metrics.get("meta_acc")
                        if dataset_metrics.get("meta_nll") is not None and not np.isnan(dataset_metrics.get("meta_nll", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_nll"] = dataset_metrics.get("meta_nll")
                        if dataset_metrics.get("meta_auc") is not None and not np.isnan(dataset_metrics.get("meta_auc", np.nan)):
                            wandb_val_final_dict[f"val/final/{dataset_key}/meta_auc"] = dataset_metrics.get("meta_auc")
                
                try:
                    final_plot_step = final_step + 1
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        self.wandb_logger.run.log(wandb_val_final_dict, step=final_step, commit=True)
                        self.wandb_logger.run.log(wandb_val_final_dict, step=final_plot_step, commit=True)
                    else:
                        self.wandb_logger.log(wandb_val_final_dict, step=final_step, commit=True)
                        self.wandb_logger.log(wandb_val_final_dict, step=final_plot_step, commit=True)
                    self.current_step = max(self.current_step, final_plot_step)
                except Exception as e:
                    raise RuntimeError(f"Error logging val/final metrics to wandb: {e}") from e
        
        # Plot wagers over time
        self._plot_wagers_over_time(wagers_history, results)
        self._plot_val_nash_gap_relationships(
            val_nash_gap_history=np.array(val_nash_gap_history, dtype=np.float32),
            val_d_regret_history=np.array(val_d_regret_history, dtype=np.float32),
            val_accuracy_history=np.array(val_accuracy_history, dtype=np.float32),
            val_history_epochs=np.array(val_nash_gap_history_epochs, dtype=np.int32),
        )
        
        return results
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint including hidden states for resuming.
        
        Note: Only saves training hidden states, not validation hidden states.
        Validation hidden states are recomputed on-demand and are not needed for resumption.
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        checkpoint = {
            "epoch": epoch,
            "wagering_method_state": self.wagering_method.state_dict(),
            "wagers_history": getattr(self, 'wagers_history', []),
            "best_d_regret": self.best_d_regret,
            "epochs_since_improvement": self.epochs_since_improvement,
        }
        
        # Save training logits if available (for resuming without recollecting)
        # if hasattr(self, 'all_model_logits') and self.all_model_logits is not None:
        #     checkpoint["all_model_logits"] = self.all_model_logits
        
        # # Do NOT save validation logits - they're recomputed on-demand during validation
        # # This saves significant disk space and memory
        
        # # Save training hidden states if available (for resuming without recollecting)
        # if hasattr(self, 'all_hidden_states') and self.all_hidden_states is not None:
        #     if isinstance(self.all_hidden_states, list):
        #         # Save as list of numpy arrays
        #         checkpoint["hidden_states"] = [hs for hs in self.all_hidden_states]
        #         checkpoint["hidden_states_format"] = "list"
        #     else:
        #         # Save as single numpy array
        #         checkpoint["hidden_states"] = self.all_hidden_states
        #         checkpoint["hidden_states_format"] = "array"
        #     log.info("Saved training hidden states to checkpoint for resuming")
        
        # Do NOT save validation hidden states - they're recomputed on-demand during batch validation
        # This saves significant disk space and memory, preventing OOM errors
        
        torch.save(checkpoint, checkpoint_path)
    
    def _plot_validation_wagers_by_dataset(
        self,
        val_wagers: np.ndarray,
        results: Dict[str, Any],
    ):
        """Plot average validation wagers grouped by dataset."""
        if "dataset_indices" not in results or self.checkpoint_dir is None:
            return
        
        dataset_indices = results["dataset_indices"]
        num_datasets = len(self.datasets)
        num_models = val_wagers.shape[1]
        
        # Resolve dataset names
        def _resolve_training_dataset_names():
            names = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names
        
        # Get model names
        model_names = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Plot average validation wagers per dataset (bar plot)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        dataset_names = _resolve_training_dataset_names()
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                mask = dataset_indices == dataset_idx
                if np.any(mask):
                    avg_wager = np.mean(val_wagers[mask, i])
                else:
                    avg_wager = 0.0
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers by Dataset (Validation)", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if self.checkpoint_dir:
            avg_save_path = self.checkpoint_dir / "validation_average_wagers_by_dataset.png"
            plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation average wagers by dataset plot to {avg_save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/average_by_dataset": wandb.Image(str(avg_save_path))})
        
        plt.close()
    
    def _plot_wagers_over_time(
        self,
        wagers_history: np.ndarray,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
    ):
        """Plot average wagers over time, overall and grouped by dataset."""
        num_examples, num_models = wagers_history.shape
        
        # Prefer human-readable model names from metadata (original config),
        # fall back to model objects' paths, and finally to generic names.
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]
        
        # If metadata is missing or length mismatch, try to infer from model objects
        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]
        
        # Final safety fallback
        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]
        
        # Plot 1: Overall wagers over time
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_steps = np.arange(1, num_examples + 1)
        
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title("Average Wagers Over Time (All Datasets)", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path is None and self.checkpoint_dir:
            save_path = self.checkpoint_dir / "wagers_over_time.png"
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved overall wagers plot to {save_path}")
            
            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/overall": wandb.Image(str(save_path))})
        
        plt.close()
        
        # Helper: resolve training dataset names for display
        def _resolve_training_dataset_names() -> List[str]:
            names: List[str] = []
            if isinstance(self.metadata, dict):
                for key in ["training_datasets", "dataset_names", "datasets", "train_datasets"]:
                    v = self.metadata.get(key)
                    if isinstance(v, (list, tuple)) and len(v) > 0:
                        names = [str(x) for x in v][: len(self.datasets)]
                        break
                    if isinstance(v, str) and len(self.datasets) == 1:
                        names = [v]
                        break
            if not names:
                inferred = []
                for i, ds in enumerate(self.datasets):
                    ds_name = getattr(ds, "name", None) or getattr(ds, "dataset_name", None) or getattr(ds, "path", None)
                    inferred.append(str(ds_name) if ds_name else f"dataset_{i}")
                names = inferred[: len(self.datasets)]
            # Final fallback
            if len(names) != len(self.datasets):
                names = [f"dataset_{i}" for i in range(len(self.datasets))]
            return names

        # Plot 2: Wagers grouped by dataset
        if "dataset_indices" in results:
            dataset_indices = results["dataset_indices"]
            num_datasets = len(self.datasets)
            dataset_names_disp = _resolve_training_dataset_names()
            
            # Create a subplot for each dataset
            fig, axes = plt.subplots(num_datasets, 1, figsize=(10, 4 * num_datasets))
            if num_datasets == 1:
                axes = [axes]
            
            for dataset_idx in range(num_datasets):
                ax = axes[dataset_idx]
                
                # Get mask for this dataset
                mask = dataset_indices == dataset_idx
                if not np.any(mask):
                    continue
                
                # Extract wagers for this dataset
                dataset_wagers = wagers_history[mask]
                dataset_steps = np.arange(1, len(dataset_wagers) + 1)
                
                # Plot wagers for each model
                for i in range(num_models):
                    ax.plot(dataset_steps, dataset_wagers[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
                
                dataset_name = dataset_names_disp[dataset_idx] if dataset_idx < len(dataset_names_disp) else f"dataset_{dataset_idx}"
                ax.set_xlabel("Training Step (within dataset)", fontsize=10)
                ax.set_ylabel("Wager (Weight)", fontsize=10)
                ax.set_title(f"Wagers Over Time - {dataset_name}", fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            
            if self.checkpoint_dir:
                grouped_save_path = self.checkpoint_dir / "wagers_over_time_by_dataset.png"
                plt.savefig(grouped_save_path, dpi=150, bbox_inches='tight')
                log.debug(f"Saved grouped wagers plot to {grouped_save_path}")
                
                if self.wandb_logger:
                    import wandb
                    self._log_wandb_plot({"wagers_plot/by_dataset": wandb.Image(str(grouped_save_path))})
            
            plt.close()
            
            # Plot 3: Average wagers per dataset (bar plot)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            dataset_names = dataset_names_disp
            x = np.arange(num_datasets)
            width = 0.8 / num_models
            
            for i in range(num_models):
                avg_wagers = []
                for dataset_idx in range(num_datasets):
                    mask = dataset_indices == dataset_idx
                    if np.any(mask):
                        avg_wager = np.mean(wagers_history[mask, i])
                    else:
                        avg_wager = 0.0
                    avg_wagers.append(avg_wager)
                
                ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
            
            ax.set_xlabel("Dataset", fontsize=11)
            ax.set_ylabel("Average Wager (Weight)", fontsize=11)
            ax.set_title("Average Wagers by Dataset", fontsize=12, fontweight='bold')
            ax.set_xticks(x + width * (num_models - 1) / 2)
            ax.set_xticklabels(dataset_names, rotation=20, ha='right')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 1.05])
            
            plt.tight_layout()
            
            if self.checkpoint_dir:
                avg_save_path = self.checkpoint_dir / "average_wagers_by_dataset.png"
                plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
                log.debug(f"Saved average wagers by dataset plot to {avg_save_path}")
                
                if self.wandb_logger:
                    import wandb
                    self._log_wandb_plot({"wagers_plot/average_by_dataset": wandb.Image(str(avg_save_path))})
            
            plt.close()

    def _plot_val_nash_gap_relationships(
        self,
        val_nash_gap_history: np.ndarray,
        val_d_regret_history: np.ndarray,
        val_accuracy_history: np.ndarray,
        val_history_epochs: np.ndarray,
    ):
        """Plot validation Nash-gap relationships against d_regret and accuracy."""
        if self.checkpoint_dir is None:
            return

        if val_nash_gap_history.size == 0:
            log.debug("Skipping val_nash_gap plots: no validation Nash-gap history available")
            return

        # Plot 1: val_nash_gap (x) vs d_regret (y)
        d_regret_mask = np.isfinite(val_nash_gap_history) & np.isfinite(val_d_regret_history)
        if np.any(d_regret_mask):
            x_vals = val_nash_gap_history[d_regret_mask]
            y_vals = val_d_regret_history[d_regret_mask]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(x_vals, y_vals, s=32, alpha=0.85)
            ax.set_xlabel("Validation Nash Gap (Mean)", fontsize=11)
            ax.set_ylabel("Validation D-Regret", fontsize=11)
            ax.set_title("Validation D-Regret vs Mean Nash Gap", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = self.checkpoint_dir / "validation_nash_gap_vs_d_regret.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation nash_gap vs d_regret plot to {save_path}")

            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/nash_gap_vs_d_regret": wandb.Image(str(save_path))})

            plt.close()
        else:
            log.debug("Skipping val_nash_gap vs d_regret plot: no finite paired points")

        # Plot 2: val_nash_gap (x) vs accuracy (y)
        accuracy_mask = np.isfinite(val_nash_gap_history) & np.isfinite(val_accuracy_history)
        if np.any(accuracy_mask):
            x_vals = val_nash_gap_history[accuracy_mask]
            y_vals = val_accuracy_history[accuracy_mask]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(x_vals, y_vals, s=32, alpha=0.85)
            ax.set_xlabel("Validation Nash Gap (Mean)", fontsize=11)
            ax.set_ylabel("Validation Accuracy", fontsize=11)
            ax.set_title("Validation Accuracy vs Mean Nash Gap", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.0, 1.05])
            plt.tight_layout()

            save_path = self.checkpoint_dir / "validation_nash_gap_vs_accuracy.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.debug(f"Saved validation nash_gap vs accuracy plot to {save_path}")

            if self.wandb_logger:
                import wandb
                self._log_wandb_plot({"wagers_plot/val/nash_gap_vs_accuracy": wandb.Image(str(save_path))})

            plt.close()
        else:
            log.debug("Skipping val_nash_gap vs accuracy plot: no finite paired points")

        if val_history_epochs.size > 0:
            log.debug(
                f"Tracked validation mean Nash gap for {val_history_epochs.size} epochs "
                f"(first epoch={int(val_history_epochs[0])}, last epoch={int(val_history_epochs[-1])})"
            )

    def _plot_val_wagers_vs_score_diff_for_epoch(
        self,
        val_wagers: np.ndarray,
        val_score_diffs: np.ndarray,
        epoch: int,
    ):
        """Scatter plot of validation wagers vs score differences for a specific epoch."""
        if self.checkpoint_dir is None:
            return

        if val_wagers is None or val_score_diffs is None:
            log.debug(f"Skipping epoch {epoch + 1} wagers vs score_diff plot: missing wagers or score_diff")
            return

        val_wagers = np.asarray(val_wagers)
        val_score_diffs = np.asarray(val_score_diffs)
        if val_wagers.ndim != 2 or val_score_diffs.ndim != 2 or val_wagers.shape != val_score_diffs.shape:
            log.debug(
                f"Skipping epoch {epoch + 1} wagers vs score_diff plot: shape mismatch "
                f"wagers={val_wagers.shape}, score_diffs={val_score_diffs.shape}"
            )
            return

        num_models = val_wagers.shape[1]

        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "models" in self.metadata:
            raw_names = self.metadata["models"]
            if isinstance(raw_names, (list, tuple)):
                model_names = [str(name) for name in raw_names][:num_models]

        if len(model_names) != num_models and getattr(self, "models", None):
            inferred_names: List[str] = []
            for i, model in enumerate(self.models):
                name = getattr(model, "model_path", None)
                if not name:
                    name = getattr(model, "model_name", None)
                if not name:
                    name = f"Model {i+1}"
                inferred_names.append(str(name))
            model_names = inferred_names[:num_models]

        if len(model_names) != num_models:
            model_names = [f"Model {i+1}" for i in range(num_models)]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plotted_any = False
        for model_idx in range(num_models):
            model_x = val_wagers[:, model_idx]
            model_y = val_score_diffs[:, model_idx]
            finite_mask = np.isfinite(model_x) & np.isfinite(model_y)
            if not np.any(finite_mask):
                continue
            ax.scatter(
                model_x[finite_mask],
                model_y[finite_mask],
                s=14,
                alpha=0.45,
                label=model_names[model_idx],
            )
            plotted_any = True

        if not plotted_any:
            plt.close()
            log.debug(f"Skipping epoch {epoch + 1} wagers vs score_diff plot: no finite points")
            return

        ax.set_xlabel("Validation Wagers (all models × val samples)", fontsize=11)
        ax.set_ylabel("Validation Score Diff", fontsize=11)
        ax.set_title(f"Validation Score Diff vs Wagers (Epoch {epoch + 1})", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = self.checkpoint_dir / f"validation_epoch_{epoch + 1:04d}_wagers_vs_score_diff.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved epoch {epoch + 1} wagers vs score_diff plot to {save_path}")

        if self.wandb_logger:
            import wandb
            self._log_wandb_plot({f"wagers_plot/val/wagers_vs_score_diff/epoch_{epoch + 1}": wandb.Image(str(save_path))})

        plt.close()
    
    def save_final_checkpoint(self, save_dir: str) -> str:
        """Save final checkpoint and return the path.
        
        Returns:
            str: Path to the saved checkpoint directory
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we save the best epoch state if available
        if self.best_wagering_method_state is not None:
            log.debug("Loading best checkpoint state before saving final checkpoint")
            self.wagering_method.load_state_dict(self.best_wagering_method_state)

        # Save wagering method (contains best epoch state if early stopping occurred)
        self.wagering_method.save_pretrained(str(save_dir))
        
        if self.best_epoch is not None:
            log.debug(f"Saved final checkpoint to {save_dir} (best epoch: {self.best_epoch + 1})")
        else:
            log.debug(f"Saved final checkpoint to {save_dir}")
        return str(save_dir)

