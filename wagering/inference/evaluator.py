"""
Inference/evaluation pipeline for multi-LLM wagering methods.
"""

import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Add src/ to path for lm_polygraph imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.ue_metrics.ece import ECE

# Local wagering imports
from wagering.methods.base import WageringMethod
from wagering.training.analytics import WageringAnalytics
from wagering.training.trainer import compute_dynamic_regret, compute_meta_metrics
from wagering.aggregation.base import AggregationFunction
from wagering.utils.multi_llm_ensemble import (
    collect_option_logits_and_hidden_states_for_model,
    get_cached_logits_and_hidden_states_for_model,
    set_cached_logits_and_hidden_states_for_model,
)

log = logging.getLogger("wagering")

from sklearn.metrics import roc_auc_score


class WageringEvaluator:
    """
    Evaluator for multi-LLM wagering methods.
    
    Evaluates on test splits and OOD datasets, computes accuracy, AUC, and ECE.
    """
    
    def __init__(
        self,
        models: List[WhiteboxModel],
        wagering_method: WageringMethod,
        aggregation_function: AggregationFunction,
        option_tokens: List[str] = ["A", "B", "C", "D"],
        wandb_logger: Optional[Any] = None,
        checkpoint_dir: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
        training_checkpoint_path: Optional[str] = None,
        seed: Optional[int] = None,
        wandb_starting_step: Optional[int] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            models: List of WhiteboxModel instances
            wagering_method: WageringMethod instance (should be in eval mode)
            aggregation_function: AggregationFunction instance
            option_tokens: List of option tokens (e.g., ["A", "B", "C", "D"])
            wandb_logger: Optional wandb logger for logging metrics
            checkpoint_dir: Optional directory to save/load evaluation checkpoints
            metadata: Optional metadata dict with model_names, training_datasets, etc.
            training_checkpoint_path: Optional path to the training checkpoint used
            seed: Optional random seed used for this run
            wandb_starting_step: Optional starting step for wandb logging (useful when resuming from training)
        """
        self.models = models
        self.wagering_method = wagering_method
        self.aggregation_function = aggregation_function
        self.option_tokens = option_tokens
        self.wandb_logger = wandb_logger
        self.checkpoint_dir = checkpoint_dir
        self.metadata = metadata or {}
        self.training_checkpoint_path = training_checkpoint_path
        self.seed = seed
        
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.wagering_method.eval_mode()
        
        # Global step counter for wandb logging across all datasets
        # This ensures each dataset gets unique step numbers and time series work correctly
        # Continue from training's last step if provided to avoid wandb step ordering warnings
        if wandb_starting_step is not None:
            # Use provided starting step (from training's last step)
            self._global_wandb_step = int(wandb_starting_step)
            run_step = self._get_wandb_run_step()
            if run_step is not None:
                self._global_wandb_step = max(self._global_wandb_step, run_step)
            log.info(f"Initialized wandb step counter to {self._global_wandb_step} (continuing from training)")
        elif self.wandb_logger:
            try:
                # Get current step from wandb run if it exists (when resuming)
                # When wandb resumes a run, we need to continue from where training left off
                if hasattr(self.wandb_logger, 'run') and self.wandb_logger.run is not None:
                    # Try to get the step from wandb's run object
                    # wandb tracks the current step internally
                    try:
                        # The step is available as wandb.run.step or can be queried from the API
                        # For resumed runs, we need to get the last logged step
                        run = self.wandb_logger.run
                        # Try accessing step directly (available in newer wandb versions)
                        if hasattr(run, 'step') and run.step is not None:
                            self._global_wandb_step = run.step
                        else:
                            # Fallback: try to get from summary (may not be available immediately)
                            # If unavailable, we'll start from 0 and wandb will handle it
                            # but this may cause warnings - better to use a high starting step
                            # Check if this is a resumed run by looking at run.id
                            # For now, start from 0 and let wandb handle step conflicts
                            self._global_wandb_step = 0
                            log.warning("Could not determine wandb step from run. Starting from 0. "
                                      "If this is a resumed run, step ordering warnings may occur.")
                    except Exception as e:
                        self._global_wandb_step = 0
                        log.warning(f"Error getting wandb step: {e}, starting from 0")
                    else:
                        if self._global_wandb_step > 0:
                            log.info(f"Initialized wandb step counter to {self._global_wandb_step} (resumed from training)")
                else:
                    self._global_wandb_step = 0
                    log.info("Initialized wandb step counter to 0 (wandb run not available)")
            except Exception as e:
                # If anything goes wrong, start from 0
                self._global_wandb_step = 0
                log.warning(f"Could not get wandb step, starting from 0: {e}")
        else:
            self._global_wandb_step = 0

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

    def _advance_wandb_step(self) -> int:
        """Advance internal wandb step while staying monotonic with run.step."""
        next_step = self._global_wandb_step + 1
        run_step = self._get_wandb_run_step()
        if run_step is not None:
            next_step = max(next_step, run_step + 1)
        self._global_wandb_step = next_step
        return self._global_wandb_step

    def _log_wandb_plot(self, payload: Dict[str, Any]) -> None:
        """Log plot payloads to wandb using a safe monotonically increasing step."""
        if not self.wandb_logger:
            return

        log_step = self._advance_wandb_step()
        if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
            self.wandb_logger.run.log(payload, step=log_step, commit=True)
        else:
            self.wandb_logger.log(payload, step=log_step, commit=True)
    
    def evaluate(
        self,
        dataset: Dataset,
        dataset_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate on a dataset.
        
        Uses shared cache to avoid recomputing logits and hidden states for the same models and datasets
        across different wagering methods. This is the default behavior since LLMs are not updated.
        
        TODO: Methods that update LLMs during evaluation should disable caching.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        log.info(f"Evaluating on {dataset_name} ({len(dataset.x)} examples)")
        
        # Check if wagering method requires hidden states (CentralizedWagers, DecentralizedWagers, or WeightedScoreWagers, etc.)
        wagering_method_name = type(self.wagering_method).__name__
        needs_hidden_states = wagering_method_name not in ["EqualWagers", "ZeroOneWagers", "OneZeroWagers"]
        
        # Check cache per model and collect if needed
        all_model_logits_list = []
        all_model_hidden_states_list = [] if needs_hidden_states else None
        labels = None
        
        for i, model in enumerate(self.models):
            # Try to load from cache for this model
            model_path = model if isinstance(model, str) else model.model_path
            cached_logits, cached_hidden_states, cached_labels = get_cached_logits_and_hidden_states_for_model(
                model_path, dataset, self.option_tokens
            )
            
            if cached_logits is not None:
                log.debug(f"Model {i+1}/{len(self.models)}: Using cached logits")
                all_model_logits_list.append(cached_logits)
                
                if needs_hidden_states:
                    if cached_hidden_states is not None:
                        log.info(f"Model {i+1}/{len(self.models)}: Using cached hidden states")
                        all_model_hidden_states_list.append(cached_hidden_states)
                    else:
                        log.info(f"Model {i+1}/{len(self.models)}: Hidden states not cached - will collect")
                        # Need to collect for this model
                        if isinstance(model, str):
                            raise RuntimeError(
                                f"Cache miss for model path {model}. Model must be loaded to collect logits."
                            )
                        model_logits, model_hidden_states, model_labels = collect_option_logits_and_hidden_states_for_model(
                            model, dataset, self.option_tokens
                        )
                        # Update cache with hidden states
                        set_cached_logits_and_hidden_states_for_model(
                            model, dataset, self.option_tokens,
                            model_logits, model_hidden_states, model_labels
                        )
                        all_model_logits_list[-1] = model_logits  # Use freshly collected logits
                        all_model_hidden_states_list.append(model_hidden_states)
                        labels = model_labels
                
                # Set labels from cache if not already set
                if labels is None:
                    labels = cached_labels
            else:
                # Cache miss - collect both logits and hidden states
                log.info(f"Model {i+1}/{len(self.models)}: Cache miss - collecting logits and hidden states")
                if isinstance(model, str):
                    raise RuntimeError(
                        f"Cache miss for model path {model}. Model must be loaded to collect logits."
                    )
                model_logits, model_hidden_states, model_labels = collect_option_logits_and_hidden_states_for_model(
                    model, dataset, self.option_tokens
                )
                
                # Cache the results for this model
                set_cached_logits_and_hidden_states_for_model(
                    model, dataset, self.option_tokens,
                    model_logits, model_hidden_states, model_labels
                )
                
                all_model_logits_list.append(model_logits)
                if needs_hidden_states:
                    all_model_hidden_states_list.append(model_hidden_states)
                labels = model_labels
        
        # Stack into final arrays
        all_model_logits = np.stack(all_model_logits_list, axis=0)  # [num_models, num_examples, num_options]
        
        if needs_hidden_states and all_model_hidden_states_list:
            # Check if all hidden states have the same shape
            hidden_dims = [hs.shape[-1] for hs in all_model_hidden_states_list]
            if len(set(hidden_dims)) == 1:
                # All same dimension - stack into single array
                all_model_hidden_states = np.stack(all_model_hidden_states_list, axis=0)
                log.info(f"Stacked hidden states: shape {all_model_hidden_states.shape}")
            else:
                # Different dimensions - keep as list
                log.info(f"Models have different hidden dimensions: {hidden_dims}. Keeping as list.")
                all_model_hidden_states = all_model_hidden_states_list
        else:
            all_model_hidden_states = None
        
        # Ensure labels are numpy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int32)
        
        # Evaluate on all examples in batches for efficiency
        all_predictions = []
        all_aggregated_probs = []
        wagers_history = []  # Track wagers for each example
        
        # Running metrics for per-step logging
        running_correct = 0
        running_nll_sum = 0.0
        
        num_examples = all_model_logits.shape[1]
        eval_batch_size = 100  # Process evaluation in batches of 100
        
        for batch_start in range(0, num_examples, eval_batch_size):
            batch_end = min(batch_start + eval_batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            # Get batch of logits
            batch_logits = all_model_logits[:, batch_start:batch_end, :]  # [num_models, batch_size, num_options]
            batch_logits_transposed = np.transpose(batch_logits, (1, 0, 2))  # [batch_size, num_models, num_options]
            batch_labels = labels[batch_start:batch_end]  # [batch_size]
            
            # Get questions for batch (for wagering methods that need them)
            batch_questions = dataset.x[batch_start:batch_end]  # List of question strings
            
            # Prepare hidden states for batch if available
            batch_hidden_states = None
            if all_model_hidden_states is not None:
                if isinstance(all_model_hidden_states, list):
                    # List of arrays with different dimensions - keep as list
                    batch_hidden_states = []
                    for i in range(len(all_model_hidden_states)):
                        model_hs = all_model_hidden_states[i][batch_start:batch_end, :]  # [batch_size, hidden_dim_i]
                        batch_hidden_states.append(model_hs)
                else:
                    # Stacked array: [num_models, num_examples, hidden_dim]
                    batch_hidden_states_array = all_model_hidden_states[:, batch_start:batch_end, :]
                    # Convert to list of [num_models] arrays, each [batch_size, hidden_dim]
                    batch_hidden_states = [batch_hidden_states_array[i, :, :] for i in range(batch_hidden_states_array.shape[0])]
            
            # Compute wagers for batch
            res_dict = self.wagering_method.compute_wagers(
                model_logits=batch_logits_transposed,
                gold_label=batch_labels,
                hidden_states_list=batch_hidden_states,
                questions=batch_questions
            )  # [batch_size, num_models]
            batch_wagers = res_dict["wagers"]
            # Aggregate predictions for batch
            batch_aggregated_log_probs, batch_aggregated_probs = self.aggregation_function.aggregate(
                batch_logits_transposed, batch_wagers
            )  # [batch_size, num_options] each
            
            batch_predictions = np.argmax(batch_aggregated_probs, axis=1)  # [batch_size]
            
            all_predictions.extend(batch_predictions.tolist())
            all_aggregated_probs.extend(batch_aggregated_probs.tolist())
            wagers_history.extend(batch_wagers.tolist())
            
            # Compute batch metrics using vectorized operations
            batch_correct = (batch_predictions == batch_labels)
            batch_nll = -np.log(batch_aggregated_probs[np.arange(batch_size_actual), batch_labels] + 1e-10)
            
            # Update running metrics
            running_correct += int(np.sum(batch_correct))
            running_nll_sum += np.sum(batch_nll)
            running_accuracy = running_correct / (batch_end)
            running_nll = running_nll_sum / batch_end
            
            # Log batch-level metrics to wandb
            if self.wandb_logger:
                log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
                
                # Log batch average metrics
                wandb_log_dict = {
                    f"{log_prefix}/{dataset_name}/batch/accuracy": float(np.mean(batch_correct)),
                    f"{log_prefix}/{dataset_name}/batch/nll": float(np.mean(batch_nll)),
                    f"{log_prefix}/{dataset_name}/batch/running_accuracy": running_accuracy,
                    f"{log_prefix}/{dataset_name}/batch/running_nll": running_nll,
                }
                
                # Add average wager statistics
                for i in range(batch_wagers.shape[1]):
                    wandb_log_dict[f"{log_prefix}/{dataset_name}/batch/wager_model_{i}"] = float(np.mean(batch_wagers[:, i]))
                
                # Use global step counter to ensure unique steps across all datasets
                log_step = self._advance_wandb_step()
                try:
                    # Use same API pattern as trainer for consistency
                    if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                        self.wandb_logger.run.log(wandb_log_dict, step=log_step)
                    else:
                        self.wandb_logger.log(wandb_log_dict, step=log_step)
                except Exception as e:
                    raise Exception(f"✗ Error logging batch metrics to wandb: {e}", exc_info=True)
        
        # Convert to arrays
        all_predictions = np.array(all_predictions, dtype=np.int32)
        all_aggregated_probs = np.stack(all_aggregated_probs, axis=0)
        wagers_history = np.stack(wagers_history, axis=0)  # [num_examples, num_models]
        
        # Compute metrics
        accuracy = np.mean(all_predictions == labels)
        
        # Compute NLL (negative log likelihood) for correct classes
        correct_class_probs = all_aggregated_probs[np.arange(len(labels)), labels]
        nll = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # Compute AUC
        auc = None
        # Use max probability as confidence score
        max_probs = all_aggregated_probs.max(axis=1)
        correctness = (all_predictions == labels).astype(int)
        
        if len(np.unique(correctness)) >= 2:
            try:
                auc = roc_auc_score(correctness, max_probs)
            except ValueError:
                log.warning("Could not compute AUC (all predictions same class)")
                auc = np.nan
        else:
            auc = np.nan
        
        # Compute ECE
        ece = None
        try:
            ece_metric = ECE(normalize=True, n_bins=20)
            # ECE expects confidence (inverted uncertainty) and target (quality metric)
            # For classification, we use max probability as confidence and correctness as target
            confidences = all_aggregated_probs.max(axis=1)
            correctness = (all_predictions == labels).astype(float)
            ece = ece_metric(confidences.tolist(), correctness.tolist())
        except Exception as e:
            log.warning(f"Could not compute ECE: {e}")
            ece = np.nan
        
        # Compute Dynamic Regret and Meta Metrics
        d_regret = None
        meta_acc = None
        meta_nll = None
        meta_auc = None
        try:
            # Get model logits in the right format [num_examples, num_models, num_options]
            model_logits = np.transpose(all_model_logits, (1, 0, 2))
            d_regret, best_expert_ids = compute_dynamic_regret(
                model_logits, all_aggregated_probs, labels
            )
            meta_metrics = compute_meta_metrics(wagers_history, best_expert_ids)
            meta_acc = meta_metrics["meta_acc"]
            meta_nll = meta_metrics["meta_nll"]
            meta_auc = meta_metrics["meta_auc"]
        except Exception as e:
            log.warning(f"Could not compute d_regret/meta metrics: {e}")
        
        results = {
            "dataset_name": dataset_name,
            "num_examples": num_examples,
            "predictions": all_predictions,
            "aggregated_probs": all_aggregated_probs,
            "labels": labels,
            "wagers_history": wagers_history,
            "accuracy": accuracy,
            "nll": nll,
            "auc": auc,
            "ece": ece,
            "d_regret": d_regret,
            "meta_acc": meta_acc,
            "meta_nll": meta_nll,
            "meta_auc": meta_auc,
        }
        
        auc_str = f"{auc:.4f}" if auc is not None and not np.isnan(auc) else "N/A"
        ece_str = f"{ece:.4f}" if ece is not None and not np.isnan(ece) else "N/A"
        d_regret_str = f"{d_regret:.4f}" if d_regret is not None and not np.isnan(d_regret) else "N/A"
        meta_acc_str = f"{meta_acc:.4f}" if meta_acc is not None and not np.isnan(meta_acc) else "N/A"
        log.info(f"{dataset_name} - Accuracy: {accuracy:.4f}, NLL: {nll:.4f}, AUC: {auc_str}, ECE: {ece_str}, "
                 f"DRegret: {d_regret_str}, MetaAcc: {meta_acc_str}")
        
        # Log average wagers per model
        avg_wagers = np.mean(wagers_history, axis=0)
        wager_info = ", ".join([f"Model {i}: {wager:.4f}" for i, wager in enumerate(avg_wagers)])
        log.info(f"{dataset_name} - Average Wagers: {wager_info}")
        
        # Create analytics dataframe for this evaluation
        training_datasets = self.metadata.get("training_datasets", [])
        if isinstance(training_datasets, str):
            training_datasets = [training_datasets]
        
        # Get dataset size (number of examples evaluated) - used to distinguish different settings
        dataset_size = len(dataset.x) if hasattr(dataset, 'x') and dataset.x is not None else None
        
        analytics_df = WageringAnalytics.create_evaluation_analytics(
            wagering_method=self.wagering_method,
            aggregation_function=self.aggregation_function,
            models=self.models,
            evaluation_dataset_name=dataset_name,
            training_datasets=training_datasets,
            results=results,
            metadata=self.metadata,
            checkpoint_path=self.training_checkpoint_path,
            seed=self.seed,
            dataset_size=dataset_size,
        )
        results["analytics_df"] = analytics_df
        
        # Save analytics dataframe to checkpoint directory
        if self.checkpoint_dir:
            analytics_path = self.checkpoint_dir / f"analytics_{dataset_name}.csv"
            analytics_df.to_csv(analytics_path, index=False)
            log.debug(f"Saved analytics dataframe to {analytics_path}")
        
        # Log final evaluation metrics to wandb separately from batch metrics
        # These represent the overall evaluation results, not per-batch metrics
        if self.wandb_logger:
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            log.debug(f"Logging final metrics to wandb: prefix={log_prefix}, dataset_name={dataset_name}")
            # Use global step counter to ensure final metrics appear after all batch metrics
            final_step = self._advance_wandb_step()
            metric_values = {
                "accuracy": accuracy,
                "nll": nll,
                "auc": auc if auc is not None and not np.isnan(auc) else None,
                "ece": ece if ece is not None and not np.isnan(ece) else None,
                "d_regret": d_regret if d_regret is not None and not np.isnan(d_regret) else None,
                "meta_acc": meta_acc if meta_acc is not None and not np.isnan(meta_acc) else None,
                "meta_nll": meta_nll if meta_nll is not None and not np.isnan(meta_nll) else None,
                "meta_auc": meta_auc if meta_auc is not None and not np.isnan(meta_auc) else None,
            }

            primary_final_prefix = f"{log_prefix}/{dataset_name}/final"
            alias_final_prefix = f"{log_prefix}/final/{dataset_name}"
            wandb_metrics = {f"{primary_final_prefix}/{k}": v for k, v in metric_values.items()}
            wandb_metrics.update({f"{alias_final_prefix}/{k}": v for k, v in metric_values.items()})
            
            # Add average wagers per model
            avg_wagers = np.mean(wagers_history, axis=0)
            for model_idx, avg_wager in enumerate(avg_wagers):
                wager_value = float(avg_wager)
                wandb_metrics[f"{primary_final_prefix}/avg_wager_model_{model_idx}"] = wager_value
                wandb_metrics[f"{alias_final_prefix}/avg_wager_model_{model_idx}"] = wager_value
            
            try:
                final_plot_step = final_step + 1
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(wandb_metrics, step=final_step, commit=True)
                    self.wandb_logger.run.log(wandb_metrics, step=final_plot_step, commit=True)
                else:
                    self.wandb_logger.log(wandb_metrics, step=final_step, commit=True)
                    self.wandb_logger.log(wandb_metrics, step=final_plot_step, commit=True)
            except Exception as e:
                raise Exception(f"Error logging final metrics to wandb: {e}", exc_info=True)
        
        # Plot wagers for this evaluation
        self._plot_evaluation_wagers(results)
        
        return results
    
    def _plot_evaluation_wagers(self, results: Dict[str, Any]):
        """
        Plot wagers during evaluation for a dataset.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from typing import List
        
        if "wagers_history" not in results or self.checkpoint_dir is None:
            return
        
        wagers_history = results["wagers_history"]
        dataset_name = results["dataset_name"]
        num_examples, num_models = wagers_history.shape
        
        # Get model names from metadata
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
        
        # Plot wagers over time
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        time_steps = np.arange(1, num_examples + 1)
        
        for i in range(num_models):
            ax.plot(time_steps, wagers_history[:, i], label=model_names[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel("Evaluation Step", fontsize=11)
        ax.set_ylabel("Wager (Weight)", fontsize=11)
        ax.set_title(f"Wagers Over Time - {dataset_name}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"wagers_over_time_{dataset_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved wagers plot to {save_path}")
        
        if self.wandb_logger:
            import wandb
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            try:
                self._log_wandb_plot({f"{log_prefix}/{dataset_name}/wagers_plot": wandb.Image(str(save_path))})
            except Exception as e:
                raise Exception(f"Error logging wagers plot: {e}")
        
        plt.close()
        
        # Plot average wagers per model
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        avg_wagers = np.mean(wagers_history, axis=0)
        
        bars = ax.bar(range(num_models), avg_wagers, alpha=0.7, color='steelblue')
        
        # Add value labels on bars
        for bar, wager in zip(bars, avg_wagers):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{wager:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title(f"Average Wagers by Model - {dataset_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        avg_save_path = self.checkpoint_dir / f"average_wagers_{dataset_name}.png"
        plt.savefig(avg_save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved average wagers plot to {avg_save_path}")
        
        if self.wandb_logger:
            import wandb
            log_prefix = "test" if not dataset_name.startswith("ood_") else "ood"
            try:
                self._log_wandb_plot({f"{log_prefix}/{dataset_name}/average_wagers_plot": wandb.Image(str(avg_save_path))})
            except Exception as e:
                raise Exception(f"Error logging average wagers plot: {e}")
        
        plt.close()
    
    def _save_checkpoint(self, all_results: Dict[str, Any], completed_datasets: List[str]):
        """Save evaluation checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint_file = self.checkpoint_dir / "eval_checkpoint.pkl"
        checkpoint_data = {
            "results": all_results,
            "completed_datasets": completed_datasets,
            "global_wandb_step": self._global_wandb_step,  # Save global step counter
        }
        
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            log.info(f"Saved evaluation checkpoint to {checkpoint_file}")
        except Exception as e:
            raise Exception(f"Failed to save checkpoint: {e}")
    
    def _plot_average_wagers_across_datasets(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        datasets_list: List[Tuple[Dataset, str]],
        eval_type: str = "test",
    ):
        """
        Plot average wagers grouped by dataset (aggregated across multiple evaluation datasets).
        
        Args:
            results_dict: Dictionary of results from evaluate() calls
            datasets_list: List of (dataset, name) tuples
            eval_type: Either "test" or "ood" for logging prefix
        """
        if self.checkpoint_dir is None or not results_dict:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Collect model names
        model_names: List[str] = []
        num_models = None
        
        # Get first result to determine number of models
        first_result = next(iter(results_dict.values())) if results_dict else None
        if first_result and "wagers_history" in first_result:
            num_models = first_result["wagers_history"].shape[1]
            
            # Try to get model names from metadata
            if isinstance(self.metadata, dict) and "model_names" in self.metadata:
                raw_names = self.metadata["model_names"]
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
        
        if num_models is None:
            log.warning("Could not determine number of models for plotting")
            return
        
        # Prepare data for plotting: average wagers per dataset
        num_datasets = len(datasets_list)
        dataset_names = [name for _, name in datasets_list]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        # For each model, compute average wager across examples in each dataset
        for model_idx in range(num_models):
            avg_wagers_per_dataset = []
            
            for dataset_name in dataset_names:
                if dataset_name in results_dict and "wagers_history" in results_dict[dataset_name]:
                    wagers_history = results_dict[dataset_name]["wagers_history"]
                    avg_wager = np.mean(wagers_history[:, model_idx])
                else:
                    avg_wager = 0.0
                avg_wagers_per_dataset.append(avg_wager)
            
            ax.bar(x + model_idx * width, avg_wagers_per_dataset, width, label=model_names[model_idx], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        ax.set_title(f"Average Wagers by Dataset ({eval_type.capitalize()})", fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"{eval_type}_average_wagers_by_dataset.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved {eval_type} average wagers by dataset plot to {save_path}")
        
        # Log to wandb
        if self.wandb_logger:
            import wandb
            try:
                if hasattr(self.wandb_logger, 'run') and hasattr(self.wandb_logger.run, 'log'):
                    self.wandb_logger.run.log(
                        {f"wagers_plot/{eval_type}/average_by_dataset": wandb.Image(str(save_path))},
                        step=self._global_wandb_step,
                    )
                else:
                    self.wandb_logger.log(
                        {f"wagers_plot/{eval_type}/average_by_dataset": wandb.Image(str(save_path))},
                        step=self._global_wandb_step,
                    )
            except Exception as e:
                raise Exception(f"Error logging plot to wandb: {e}")
        
        plt.close()
    
    def _plot_average_wagers_across_datasets(
        self,
        all_results: Dict[str, Any],
        eval_type: str = "test",
    ):
        """
        Plot average wagers across multiple test/OOD datasets.
        
        Args:
            all_results: Dictionary with results from all datasets, keyed by dataset name
            eval_type: Either "test", "ood", or "test_and_ood" for logging prefix and filtering
        """

        if self.checkpoint_dir is None or not all_results:
            log.warning(f"SKIPPING PLOT: checkpoint_dir={self.checkpoint_dir}, all_results={bool(all_results)}")
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from typing import List
        
        # Collect wagers history and dataset names from all results
        dataset_names = []
        all_wagers_list = []
        
        for dataset_name, result in all_results.items():
            # Filter based on eval_type
            is_ood = dataset_name.startswith("ood_")
            if eval_type == "test" and is_ood:
                continue
            elif eval_type == "ood" and not is_ood:
                continue
            # For "test_and_ood", include all datasets (no filtering)
            
            if "wagers_history" in result:
                wagers = result["wagers_history"]  # [num_examples, num_models]
                all_wagers_list.append(wagers)
                # Clean up dataset name for display
                display_name = dataset_name.replace("ood_", "")
                if is_ood and eval_type == "test_and_ood":
                    display_name = f"[OOD] {display_name}"
                dataset_names.append(display_name)
        
        if not all_wagers_list:
            log.warning(f"No wagers history found for {eval_type} evaluation")
            return
        
        # Compute average wagers per dataset
        num_datasets = len(all_wagers_list)
        num_models = all_wagers_list[0].shape[1]
        
        # Get model names
        model_names: List[str] = []
        if isinstance(self.metadata, dict) and "model_names" in self.metadata:
            raw_names = self.metadata["model_names"]
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
        
        # Plot: Average wagers per dataset
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(num_datasets)
        width = 0.8 / num_models
        
        for i in range(num_models):
            avg_wagers = []
            for dataset_idx in range(num_datasets):
                wagers = all_wagers_list[dataset_idx]
                avg_wager = np.mean(wagers[:, i])
                avg_wagers.append(avg_wager)
            
            ax.bar(x + i * width, avg_wagers, width, label=model_names[i], alpha=0.8)
        
        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Average Wager (Weight)", fontsize=11)
        
        # Set title based on eval_type
        if eval_type == "test":
            title = "Average Wagers by Dataset (Test)"
        elif eval_type == "ood":
            title = "Average Wagers by Dataset (OOD)"
        else:  # test_and_ood
            title = "Average Wagers by Dataset (Test + OOD)"
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * (num_models - 1) / 2)
        ax.set_xticklabels(dataset_names, rotation=20, ha='right')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.checkpoint_dir / f"average_wagers_by_dataset_{eval_type}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.debug(f"Saved wagers plot ({eval_type}) to {save_path}")
        
        if self.wandb_logger:
            import wandb
            plot_image_primary = wandb.Image(str(save_path))
            plot_image_alias = wandb.Image(str(save_path))
            self._log_wandb_plot(
                {
                    f"wagers_plot/{eval_type}/average_by_dataset": plot_image_primary,
                    f"wagers_plot/average_by_dataset/{eval_type}": plot_image_alias,
                }
            )
        
        plt.close(fig)
        
        plt.close()
    
    def evaluate_multiple(
        self,
        test_datasets: List[Tuple[Dataset, str]],
        ood_dataset: Optional[Tuple[Dataset, str]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on multiple test datasets and optionally an OOD dataset.
        
        Args:
            test_datasets: List of (dataset, name) tuples for test splits
            ood_dataset: Optional (dataset, name) tuple for OOD evaluation
            resume: If True, attempt to resume from checkpoint if available (DISABLED - always evaluates from scratch)
            
        Returns:
            Dictionary with evaluation results for all datasets, including a combined analytics_df
        """
        all_results = {}
        completed_datasets = []
        all_analytics_dfs = []
        
        # Evaluate on test splits
        for dataset, name in test_datasets:
            log.info(f"Evaluating test dataset: {name}")
            results = self.evaluate(dataset, name)
            all_results[name] = results
            completed_datasets.append(name)
            
            # Collect analytics dataframe
            if "analytics_df" in results:
                all_analytics_dfs.append(results["analytics_df"])
            
            # Save checkpoint after each dataset
            # self._save_checkpoint(all_results, completed_datasets)
        
        # Evaluate on OOD dataset if provided
        if ood_dataset is not None:
            dataset, name = ood_dataset
            ood_name = f"ood_{name}"
            
            log.info(f"Evaluating OOD dataset: {name} -> {ood_name}")
            log.info(f"Wandb logger available: {self.wandb_logger is not None}")
            
            # Pass ood_name to evaluate() so it gets the correct "ood" prefix in wandb logging
            results = self.evaluate(dataset, ood_name)
            all_results[ood_name] = results
            completed_datasets.append(ood_name)
            
            # Collect analytics dataframe
            if "analytics_df" in results:
                all_analytics_dfs.append(results["analytics_df"])
            
            # Save checkpoint after OOD evaluation
            # self._save_checkpoint(all_results, completed_datasets)
        
        # Combine all analytics dataframes and save
        if all_analytics_dfs and self.checkpoint_dir:
            combined_analytics = pd.concat(all_analytics_dfs, ignore_index=True)
            combined_path = self.checkpoint_dir / "analytics_all.csv"
            combined_analytics.to_csv(combined_path, index=False)
            log.debug(f"Saved combined analytics dataframe to {combined_path}")
            all_results["analytics_df"] = combined_analytics
        
        # Plot average wagers across all test datasets
        log.info("=== GENERATING PLOTS ===")
        log.debug(f"All results keys: {list(all_results.keys())}")
        
        log.info("Generating test datasets plot...")
        self._plot_average_wagers_across_datasets(all_results, "test")
        
        # Plot average wagers across OOD datasets if applicable
        if ood_dataset is not None:
            # Create filtered results dict with only OOD datasets
            ood_results = {k: v for k, v in all_results.items() if k.startswith("ood_")}
            if ood_results:
                log.info("Generating OOD datasets plot...")
                self._plot_average_wagers_across_datasets(ood_results, "ood")
        
        # Plot average wagers across both test and OOD datasets combined
        log.info("Generating combined test+OOD plot...")
        self._plot_average_wagers_across_datasets(all_results, "test_and_ood")
        log.info("=== PLOTS GENERATION COMPLETE ===")
        
        return all_results

