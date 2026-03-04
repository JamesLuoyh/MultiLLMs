"""
Multi-LLM ensemble utilities for wagering package.

Contains functions for collecting logits and hidden states from models,
with disk-based caching for efficiency.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Import from lm_polygraph for model and dataset classes only
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.dataset import Dataset

log = logging.getLogger("wagering")

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend by default
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("matplotlib not available; plotting functions will be disabled")

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("sklearn not available; AUC calculation will be disabled")


@dataclass
class LogitCacheKey:
    """
    Lightweight identifier for a particular (dataset, split, model, size) logit run.
    """

    dataset_name: str
    split: str
    model_id: str
    num_examples: int | None = None

    def to_filename(self) -> str:
        safe_model = self.model_id.replace("/", "_")
        safe_dataset = self.dataset_name.replace("/", "_")
        safe_split = self.split.replace("/", "_")
        size_suffix = "" if self.num_examples is None else f"_{self.num_examples}"
        return f"{safe_dataset}{size_suffix}__{safe_split}__{safe_model}.npz"


class LogitCache:
    """
    Small utility for saving/loading per-option logits for multiple-choice QA.

    Stored format (npz):
        - logits: float32 array of shape [num_examples, num_options]
        - labels: int32 array of shape [num_examples]
        - meta:   UTF-8 encoded JSON string with any additional fields (optional)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: LogitCacheKey) -> Path:
        return self.cache_dir / key.to_filename()

    def save(
        self,
        key: LogitCacheKey,
        logits: np.ndarray,
        labels: np.ndarray,
        meta: Optional[Dict] = None,
    ) -> Path:
        path = self.path_for(key)
        meta = meta or {}
        np.savez_compressed(
            path,
            logits=logits.astype(np.float32),
            labels=labels.astype(np.int32),
            meta=np.string_(repr(meta)),
        )
        log.info(
            f"Saved logits cache for dataset={key.dataset_name}, "
            f"split={key.split}, model={key.model_id} to {path}"
        )
        return path

    def load(self, key: LogitCacheKey) -> Tuple[np.ndarray, np.ndarray]:
        path = self.path_for(key)
        if not path.exists():
            raise FileNotFoundError(f"Logit cache not found at {path}")
        data = np.load(path, allow_pickle=True)
        logits = data["logits"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        return logits, labels


# Disk-based cache directory for logits and hidden states
_WAGERING_CACHE_DIR = Path("/common/users/yl2310/MultiLLMs/wagering_model_logits_states_caches")
_WAGERING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_model_path_key(model: WhiteboxModel) -> str:
    """Create a cache key from a single model path."""
    return model.model_path


def _get_dataset_signature(dataset: Dataset) -> Tuple:
    """Create a dataset signature for caching based on dataset size and content hash."""
    dataset_size = len(dataset.x)
    # Create a hash from first 3 examples for uniqueness
    sample_text = "\n".join(dataset.x[:min(3, len(dataset.x))]) if dataset.x else ""
    content_hash = hashlib.md5(sample_text.encode('utf-8')).hexdigest()[:8]
    return (dataset_size, content_hash)


def _cache_key_to_filename(cache_key: Tuple) -> str:
    """Convert a cache key to a filename-safe string using MD5 hash."""
    key_str = json.dumps(cache_key, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
    return f"{key_hash}.npz"


def _get_cache_path(cache_key: Tuple) -> Path:
    """Get the file path for a cache key."""
    filename = _cache_key_to_filename(cache_key)
    return _WAGERING_CACHE_DIR / filename


def get_cached_logits_and_hidden_states_for_model(
    model_path: str,
    dataset: Dataset,
    option_tokens: List[str],
) -> Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Get cached logits and hidden states for a model if available from disk.

    Args:
        model_path: Model path string
        dataset: Dataset instance
        option_tokens: List of option tokens (e.g., ['A', 'B', 'C', 'D'])

    Returns:
        Tuple of (logits, hidden_states, labels) if cached, else (None, None, None)
        logits shape: [num_examples, num_options]
        hidden_states shape: [num_examples, hidden_dim]
        labels shape: [num_examples]
    """
    model_key = model_path
    dataset_key = _get_dataset_signature(dataset)
    option_key = tuple(option_tokens)
    cache_key = (model_key, dataset_key, option_key)
    cache_path = _get_cache_path(cache_key)

    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=True)
            logits = data["logits"] if "logits" in data else None
            hidden_states = data["hidden_states"] if "hidden_states" in data else None
            labels = data["labels"] if "labels" in data else None

            # Handle hidden_states if it was pickled
            if "hidden_states_pickle" in data:
                hidden_states = pickle.loads(data["hidden_states_pickle"].item())

            log.debug(f"Cache hit for model {model_path} and dataset size {len(dataset.x)}")
            return logits, hidden_states, labels
        except Exception as e:
            raise Exception(f"Error loading cache from {cache_path}: {e}")
    return None, None, None


def set_cached_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    logits: Optional[np.ndarray],
    hidden_states: Optional[np.ndarray],
    labels: Optional[np.ndarray],
):
    """
    Cache logits and hidden states for a single model on disk.
    
    Args:
        model: WhiteboxModel instance
        dataset: Dataset instance
        option_tokens: List of option tokens
        logits: Optional np.ndarray of shape [num_examples, num_options]
        hidden_states: Optional np.ndarray of shape [num_examples, hidden_dim]
        labels: Optional np.ndarray of shape [num_examples]
    """
    model_key = _get_model_path_key(model)
    dataset_key = _get_dataset_signature(dataset)
    option_key = tuple(option_tokens)
    cache_key = (model_key, dataset_key, option_key)
    cache_path = _get_cache_path(cache_key)
    
    # Load existing cache entry if present
    existing_data = {}
    if cache_path.exists():
        try:
            existing = np.load(cache_path, allow_pickle=True)
            if "logits" in existing:
                existing_data["logits"] = existing["logits"]
            if "hidden_states" in existing:
                existing_data["hidden_states"] = existing["hidden_states"]
            if "hidden_states_pickle" in existing:
                existing_data["hidden_states"] = pickle.loads(existing["hidden_states_pickle"].item())
            if "labels" in existing:
                existing_data["labels"] = existing["labels"]
        except Exception as e:
            log.warning(f"Corrupted cache at {cache_path}: {e}. Deleting and rebuilding.")
            try:
                cache_path.unlink(missing_ok=True)
            except Exception as delete_err:
                log.warning(f"Failed to delete corrupted cache file {cache_path}: {delete_err}")
            existing_data = {}
    # Merge with existing cache entry
    cache_dict = {}
    if logits is not None:
        cache_dict["logits"] = logits.copy() if isinstance(logits, np.ndarray) else logits
    elif "logits" in existing_data:
        cache_dict["logits"] = existing_data["logits"]
    
    if hidden_states is not None:
        cache_dict["hidden_states"] = hidden_states
    elif "hidden_states" in existing_data:
        cache_dict["hidden_states"] = existing_data["hidden_states"]
    
    if labels is not None:
        cache_dict["labels"] = labels.copy() if isinstance(labels, np.ndarray) else labels
    elif "labels" in existing_data:
        cache_dict["labels"] = existing_data["labels"]
    
    # Save to disk
    try:
        save_dict = {}
        if "logits" in cache_dict and cache_dict["logits"] is not None:
            save_dict["logits"] = cache_dict["logits"].astype(np.float32) if isinstance(cache_dict["logits"], np.ndarray) else cache_dict["logits"]
        if "labels" in cache_dict and cache_dict["labels"] is not None:
            save_dict["labels"] = cache_dict["labels"].astype(np.int32) if isinstance(cache_dict["labels"], np.ndarray) else cache_dict["labels"]
        if "hidden_states" in cache_dict and cache_dict["hidden_states"] is not None:
            # Handle different data types for hidden_states
            if isinstance(cache_dict["hidden_states"], list):
                save_dict["hidden_states_pickle"] = np.void(pickle.dumps(cache_dict["hidden_states"]))
            else:
                save_dict["hidden_states"] = cache_dict["hidden_states"].astype(np.float32) if isinstance(cache_dict["hidden_states"], np.ndarray) else cache_dict["hidden_states"]
        
        np.savez_compressed(cache_path, **save_dict)
        
        items_cached = []
        if logits is not None:
            items_cached.append("logits")
        if hidden_states is not None:
            items_cached.append("hidden_states")
        if labels is not None:
            items_cached.append("labels")
        log.info(f"Cached {', '.join(items_cached) if items_cached else 'data'} for model {model.model_path} and dataset size {len(dataset.x)} to {cache_path}")
    except Exception as e:
        raise Exception(f"Error saving cache to {cache_path}: {e}", exc_info=True)


def shuffle_cached_arrays(
    array: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Shuffle a cached array using the provided indices.
    
    Args:
        array: np.ndarray to shuffle (any shape, first dimension is samples)
        indices: np.ndarray of indices to use for shuffling
        
    Returns:
        Shuffled array
    """
    return array[indices]


def filter_dataset_by_token_length(
    dataset: Dataset,
    max_prompt_tokens: int,
    tokenizer=None,
) -> Dataset:
    """
    Filter a Dataset to remove samples that exceed max_prompt_tokens.
    
    Args:
        dataset: Dataset to filter
        max_prompt_tokens: Maximum number of tokens allowed per prompt
        tokenizer: Optional tokenizer to use for counting tokens
        
    Returns:
        Filtered Dataset
    """
    if max_prompt_tokens is None or max_prompt_tokens <= 0:
        return dataset
    
    log.info(f"Filtering dataset: dropping samples longer than {max_prompt_tokens} tokens")
    kept_x, kept_y = [], []
    kept_images = [] if dataset.images is not None else None
    
    for idx, (text, target) in enumerate(zip(dataset.x, dataset.y)):
        if tokenizer is None:
            token_count = len(text.split())
        else:
            token_count = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
        
        if token_count <= max_prompt_tokens:
            kept_x.append(text)
            kept_y.append(target)
            if kept_images is not None:
                kept_images.append(dataset.images[idx])
    
    dropped = len(dataset.x) - len(kept_x)
    if dropped > 0:
        log.info(f"Dropped {dropped} samples due to length > {max_prompt_tokens} tokens")
    
    # Create new Dataset with filtered data
    filtered_dataset = Dataset(
        kept_x, 
        kept_y, 
        dataset.batch_size, 
        images=kept_images
    )
    
    return filtered_dataset


def _resolve_option_token_ids(
    model: WhiteboxModel, option_tokens: List[str], sample_prompt: str = None
) -> List[int]:
    """
    Resolve single-token IDs for answer option strings (e.g., 'A', 'B', 'C', 'D').

    Args:
        model: WhiteboxModel instance
        option_tokens: List of option strings (e.g., ['A', 'B', 'C', 'D'])
        sample_prompt: A sample prompt from the dataset (optional)

    Returns:
        List of token IDs, one per option token

    Raises:
        ValueError: If an option doesn't map to a single token in context
    """
    token_ids: List[int] = []
    
    # Determine prompt suffix
    if sample_prompt is None:
        prompt_suffix = "Answer: "
    else:
        lines = sample_prompt.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if ':' in last_line:
                prompt_suffix = last_line.rsplit(':', 1)[0] + ': '
            else:
                prompt_suffix = last_line + ' ' if last_line else "Answer: "
        else:
            prompt_suffix = "Answer: "
    
    base_prompt = sample_prompt if sample_prompt else prompt_suffix
    
    # Check if we need to handle chat template
    use_chat_template = (
        model.instruct
        and hasattr(model.tokenizer, 'chat_template')
        and model.tokenizer.chat_template is not None
    )
    
    if use_chat_template:
        try:
            chat = [{"role": "user", "content": base_prompt}]
            formatted_base = model.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
            base_ids = model.tokenizer.encode(formatted_base, add_special_tokens=False)
            
            for opt in option_tokens:
                formatted_with_opt = formatted_base + opt
                test_ids = model.tokenizer.encode(formatted_with_opt, add_special_tokens=False)
                opt_token_ids = test_ids[len(base_ids):]
                
                if len(opt_token_ids) == 1:
                    token_ids.append(opt_token_ids[0])
                elif len(opt_token_ids) > 1:
                    log.warning(
                        f"Option '{opt}' spans {len(opt_token_ids)} tokens. Using first token."
                    )
                    token_ids.append(opt_token_ids[0])
                else:
                    ids = model.tokenizer.encode(opt, add_special_tokens=False)
                    if len(ids) != 1:
                        raise ValueError(f"Option '{opt}' could not be resolved in context.")
                    token_ids.append(ids[0])
                    log.warning(f"Using standalone tokenization for '{opt}' as fallback.")
        except (ValueError, TypeError) as e:
            log.warning(f"Chat template failed: {e}. Falling back to plain tokenization.")
            use_chat_template = False
    
    if not use_chat_template:
        tokenized_base = model.tokenize([base_prompt])
        base_ids = tokenized_base['input_ids'][0].tolist()
        
        for opt in option_tokens:
            test_prompt = base_prompt + opt
            tokenized_test = model.tokenize([test_prompt])
            test_ids = tokenized_test['input_ids'][0].tolist()
            
            if len(test_ids) > len(base_ids) and test_ids[:len(base_ids)] == base_ids:
                opt_token_ids = test_ids[len(base_ids):]
                # opt_token_ids is guaranteed to have at least 1 element here
                if len(opt_token_ids) == 1:
                    token_ids.append(opt_token_ids[0])
                else:  # len(opt_token_ids) > 1
                    # log.warning(f"Option '{opt}' spans {len(opt_token_ids)} tokens. Using first token.")
                    raise ValueError(f"Option '{opt}' spans multiple tokens.")
                    # token_ids.append(opt_token_ids[0])
            else:
                # Fallback: context extraction failed, use standalone tokenization
                ids = model.tokenizer.encode(opt, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Option '{opt}' could not be resolved in context.")
                token_ids.append(ids[0])
                log.warning(f"Using standalone tokenization for '{opt}' as fallback.")
    log.info(
        f"Resolved option token IDs for {getattr(model.tokenizer, 'name_or_path', 'unknown')}: "
        f"{dict(zip(option_tokens, token_ids))}"
    )
    
    return token_ids


def collect_option_logits_and_hidden_states_for_model(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    max_new_tokens: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect per-option log-probabilities AND hidden states for a model in a single forward pass.
    
    Args:
        model: WhiteboxModel instance
        dataset: Dataset instance
        option_tokens: List of option tokens (e.g., ['A', 'B', 'C', 'D'])
        max_new_tokens: Maximum number of tokens to generate (default: 1)
        
    Returns:
        logits: np.ndarray, shape [num_examples, num_options]
        hidden_states: np.ndarray, shape [num_examples, hidden_dim]
        labels: np.ndarray, shape [num_examples]
    """
    model_device = model.device()

    if torch.cuda.is_available() and getattr(model_device, "type", None) == "cuda":
        try:
            torch.cuda.set_device(model_device)
        except Exception as e:
            raise RuntimeError(f"Could not set CUDA device to {model_device}: {e}") from e
    
    sample_prompt = dataset.x[0] if len(dataset.x) > 0 else None
    option_token_ids = _resolve_option_token_ids(model, option_tokens, sample_prompt=sample_prompt)
    
    all_log_probs: List[np.ndarray] = []
    all_hidden_states: List[np.ndarray] = []
    all_labels: List[int] = []
    
    if len(dataset.x) == 0:
        raise ValueError("Dataset is empty (0 examples).")
    
    for batch_x, batch_y, _ in dataset:
        batch = model.tokenize(batch_x)
        batch = {k: v.to(model_device) for k, v in batch.items()}
        
        generation = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        
        scores: List[torch.Tensor] = generation.scores
        if not scores:
            raise RuntimeError("Model.generate returned no scores.")
        first_step_scores = scores[0]
        
        with torch.no_grad():
            batch_log_probs = torch.stack(
                [first_step_scores[:, tid] for tid in option_token_ids],
                dim=-1,
            )
        
        # Extract hidden states
        if hasattr(generation, 'hidden_states') and generation.hidden_states is not None:
            try:
                if len(generation.hidden_states) > 1:
                    first_gen_hidden = generation.hidden_states[1]
                    if isinstance(first_gen_hidden, tuple):
                        last_layer_hidden = first_gen_hidden[-1]
                    else:
                        last_layer_hidden = first_gen_hidden
                    
                    if last_layer_hidden.dim() == 3:
                        last_token_hidden = last_layer_hidden[:, -1, :]
                    elif last_layer_hidden.dim() == 2:
                        last_token_hidden = last_layer_hidden
                    else:
                        raise ValueError(f"Unexpected hidden state shape: {last_layer_hidden.shape}")
                else:
                    input_hidden = generation.hidden_states[0]
                    if isinstance(input_hidden, tuple):
                        last_layer_hidden = input_hidden[-1]
                    else:
                        last_layer_hidden = input_hidden
                    
                    if last_layer_hidden.dim() == 3:
                        last_token_hidden = last_layer_hidden[:, -1, :]
                    elif last_layer_hidden.dim() == 2:
                        last_token_hidden = last_layer_hidden
                    else:
                        raise ValueError(f"Unexpected hidden state shape: {last_layer_hidden.shape}")
            except Exception as e:
                raise RuntimeError(
                    f"Could not extract hidden states from model {model.model_path}: {e}"
                ) from e
        else:
            raise RuntimeError(
                f"Model {model.model_path} did not return hidden_states."
            )
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                log.warning(f"CUDA synchronization error: {e}")
                raise
        
        if len(all_log_probs) == 0:
            log.info(f"Extracting logits for option tokens: {dict(zip(option_tokens, option_token_ids))}")
            token_names = {
                opt: model.tokenizer.convert_ids_to_tokens([tid])[0]
                for opt, tid in zip(option_tokens, option_token_ids)
            }
            log.info(f"Token names: {token_names}")
        
        all_log_probs.append(batch_log_probs.cpu().numpy())
        all_hidden_states.append(last_token_hidden.cpu().numpy())
        
        del generation
        del scores
        del first_step_scores
        del batch_log_probs
        del last_token_hidden
        
        if len(all_log_probs) % 10 == 0 and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                log.warning(f"CUDA error during cache clearing: {e}")
                try:
                    torch.cuda.synchronize()
                except:
                    pass
        
        for y in batch_y:
            if isinstance(y, str):
                idx = option_tokens.index(y)
            else:
                idx = int(y)
            all_labels.append(idx)
    
    if len(all_log_probs) == 0:
        raise ValueError("No batches were processed.")
    
    logits = np.concatenate(all_log_probs, axis=0)
    hidden_states = np.concatenate(all_hidden_states, axis=0)
    labels = np.asarray(all_labels, dtype=np.int32)
    return logits, hidden_states, labels


def aggregate_logits_log_pooling(
    llm_logits: np.ndarray,
    wagers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted log pooling across multiple LLMs.

    Args:
        llm_logits: np.ndarray of shape [num_llms, num_options]
        wagers: np.ndarray of shape [num_llms]

    Returns:
        pooled_log_probs: np.ndarray of shape [num_options]
        pooled_probs: np.ndarray of shape [num_options]
    """
    llm_logits = np.asarray(llm_logits, dtype=np.float32)
    wagers = np.asarray(wagers, dtype=np.float32)
    if llm_logits.ndim != 2:
        raise ValueError(f"llm_logits must have shape [num_llms, num_options], got {llm_logits.shape}")
    if wagers.ndim != 1 or wagers.shape[0] != llm_logits.shape[0]:
        raise ValueError(f"wagers shape mismatch")

    max_logits = np.max(llm_logits, axis=1, keepdims=True)
    stabilized = llm_logits - max_logits
    log_norm = max_logits + np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
    log_probs = llm_logits - log_norm

    weighted_log = wagers[:, None] * log_probs
    pooled_log_unnorm = weighted_log.sum(axis=0)

    max_pooled = np.max(pooled_log_unnorm)
    stabilized_pooled = pooled_log_unnorm - max_pooled
    log_z = max_pooled + np.log(np.exp(stabilized_pooled).sum())
    pooled_log_probs = pooled_log_unnorm - log_z
    pooled_probs = np.exp(pooled_log_probs)
    return pooled_log_probs, pooled_probs


def update_wagers(
    llm_probs: np.ndarray,
    gold_answer: int,
    current_wagers: np.ndarray,
    state: Optional[Dict] = None,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Oracle-style wager update rule for multi-LLM ensembles.

    Args:
        llm_probs: np.ndarray, shape [num_llms, num_options]
        gold_answer: int
        current_wagers: np.ndarray, shape [num_llms]
        state: Optional user-defined dictionary

    Returns:
        new_wagers: np.ndarray
        new_state: Optional[Dict]
    """
    llm_probs = np.asarray(llm_probs, dtype=np.float64)
    current_wagers = np.asarray(current_wagers, dtype=np.float64)

    if llm_probs.ndim != 2:
        raise ValueError(f"llm_probs must have shape [num_llms, num_options], got {llm_probs.shape}")
    num_llms, num_options = llm_probs.shape
    if current_wagers.shape != (num_llms,):
        raise ValueError(f"current_wagers shape mismatch")
    if not (0 <= gold_answer < num_options):
        raise ValueError(f"gold_answer out of bounds")

    k = int(gold_answer)
    w_i = current_wagers
    p_i_k = llm_probs[:, k]
    w_i_k = w_i * p_i_k

    eps = 1e-12
    w_i_k_safe = np.clip(w_i_k, eps, None)
    p_i_k_safe = np.clip(p_i_k, eps, None)

    sum_w_j_k = np.sum(w_i_k_safe)
    if sum_w_j_k <= 0.0:
        return current_wagers.astype(np.float32), state

    log_p_i_k = np.log(p_i_k_safe)
    mean_log_term = float(np.sum(log_p_i_k * w_i_k_safe) / sum_w_j_k)

    delta = w_i_k_safe * (1.0 + log_p_i_k - mean_log_term)
    new_wagers = w_i + delta - w_i_k_safe

    new_wagers = np.maximum(new_wagers, 0.0)

    return new_wagers.astype(np.float32), state


def run_online_ensemble(
    all_model_logits: List[np.ndarray],
    labels: np.ndarray,
    initial_wagers: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Simple online replay engine over pre-computed logits.

    Returns:
        Dictionary with:
            - pooled_probs: [num_examples, num_options]
            - pooled_pred:  [num_examples]
            - labels:       [num_examples]
            - wagers_history: [num_examples + 1, num_models]
    """
    num_models = len(all_model_logits)
    if num_models == 0:
        raise ValueError("all_model_logits must contain at least one model.")
    num_examples, num_options = all_model_logits[0].shape
    for idx, arr in enumerate(all_model_logits):
        if arr.shape != (num_examples, num_options):
            raise ValueError(f"Model {idx} logits shape mismatch")

    if initial_wagers is None:
        wagers = np.ones(num_models, dtype=np.float32) / float(num_models)
    else:
        wagers = np.asarray(initial_wagers, dtype=np.float32)
        if wagers.shape != (num_models,):
            raise ValueError(f"initial_wagers shape mismatch")

    pooled_probs_all = np.zeros((num_examples, num_options), dtype=np.float32)
    pooled_pred_all = np.zeros((num_examples,), dtype=np.int32)
    wagers_history = np.zeros((num_examples + 1, num_models), dtype=np.float32)
    wagers_history[0] = wagers
    state: Optional[Dict] = None

    # Pre-compute per-model probabilities
    model_probs = [None] * num_models
    for i in range(num_models):
        logits_i = all_model_logits[i].astype(np.float32)
        max_i = np.max(logits_i, axis=1, keepdims=True)
        stabilized_i = logits_i - max_i
        log_z_i = max_i + np.log(np.exp(stabilized_i).sum(axis=1, keepdims=True))
        model_probs[i] = np.exp(logits_i - log_z_i)

    for t in range(num_examples):
        llm_logits_t = np.stack([all_model_logits[i][t] for i in range(num_models)], axis=0)
        pooled_log_t, pooled_probs_t = aggregate_logits_log_pooling(llm_logits_t, wagers)
        pooled_probs_all[t] = pooled_probs_t
        pooled_pred_all[t] = int(np.argmax(pooled_probs_t))

        llm_probs_t = np.stack([model_probs[i][t] for i in range(num_models)], axis=0)
        wagers, state = update_wagers(
            llm_probs=llm_probs_t,
            gold_answer=int(labels[t]),
            current_wagers=wagers,
            state=state,
        )
        wagers_history[t + 1] = wagers

    return {
        "pooled_probs": pooled_probs_all,
        "pooled_pred": pooled_pred_all,
        "labels": labels.astype(np.int32),
        "wagers_history": wagers_history,
    }


def _calculate_cumulative_accuracy(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculate cumulative accuracy over time."""
    num_examples = len(predictions)
    cumulative_accuracy = np.zeros(num_examples, dtype=np.float32)
    
    correct_count = 0
    for t in range(num_examples):
        if predictions[t] == labels[t]:
            correct_count += 1
        cumulative_accuracy[t] = correct_count / (t + 1)
    
    return cumulative_accuracy


def _calculate_cumulative_auc(
    probs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Calculate cumulative AUC over time using max probability as confidence."""
    if not SKLEARN_AVAILABLE:
        log.warning("sklearn not available; AUC calculation skipped")
        return np.full(len(predictions), np.nan)
    
    num_examples = len(predictions)
    cumulative_auc = np.zeros(num_examples, dtype=np.float32)
    
    max_probs = probs.max(axis=1)
    correctness = (predictions == labels).astype(int)
    
    for t in range(num_examples):
        if t < 1:
            cumulative_auc[t] = np.nan
        else:
            try:
                correctness_t = correctness[:t+1]
                max_probs_t = max_probs[:t+1]
                
                if len(np.unique(correctness_t)) < 2:
                    cumulative_auc[t] = np.nan
                else:
                    auc_value = roc_auc_score(correctness_t, max_probs_t)
                    cumulative_auc[t] = auc_value
            except ValueError:
                cumulative_auc[t] = np.nan
    
    return cumulative_auc


def plot_accuracy_and_auc_over_time(
    all_model_logits: List[np.ndarray],
    ensemble_result: Dict[str, np.ndarray],
    model_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """Plot accuracy and AUC over time for individual models and ensemble."""
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available; skipping plot generation")
        return
    
    required_keys = ["labels", "pooled_probs", "pooled_pred"]
    missing_keys = [key for key in required_keys if key not in ensemble_result]
    if missing_keys:
        raise ValueError(f"ensemble_result missing required keys: {missing_keys}")
    
    labels = ensemble_result["labels"]
    pooled_probs = ensemble_result["pooled_probs"]
    pooled_pred = ensemble_result["pooled_pred"]
    
    num_examples = len(labels)
    num_models = len(all_model_logits)
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]
    
    # Calculate per-model predictions and probabilities
    model_predictions = []
    model_probs = []
    for i, logits_i in enumerate(all_model_logits):
        logits_i = logits_i.astype(np.float32)
        max_i = np.max(logits_i, axis=1, keepdims=True)
        stabilized_i = logits_i - max_i
        log_z_i = max_i + np.log(np.exp(stabilized_i).sum(axis=1, keepdims=True))
        probs_i = np.exp(logits_i - log_z_i)
        pred_i = probs_i.argmax(axis=1)
        model_probs.append(probs_i)
        model_predictions.append(pred_i)
    
    # Calculate cumulative metrics
    model_accuracies = []
    model_aucs = []
    for i in range(num_models):
        acc = _calculate_cumulative_accuracy(model_predictions[i], labels)
        auc = _calculate_cumulative_auc(model_probs[i], model_predictions[i], labels)
        model_accuracies.append(acc)
        model_aucs.append(auc)
    
    ensemble_accuracy = _calculate_cumulative_accuracy(pooled_pred, labels)
    ensemble_auc = _calculate_cumulative_auc(pooled_probs, pooled_pred, labels)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    time_steps = np.arange(1, num_examples + 1)
    
    # Plot accuracy
    for i, (acc, name) in enumerate(zip(model_accuracies, model_names)):
        ax1.plot(time_steps, acc, label=name, alpha=0.7, linewidth=1.5)
    ax1.plot(time_steps, ensemble_accuracy, label="Ensemble", linewidth=2, linestyle='--', color='black')
    ax1.set_xlabel("Number of examples seen", fontsize=11)
    ax1.set_ylabel("Cumulative Accuracy", fontsize=11)
    ax1.set_title("Accuracy Over Time", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot AUC
    for i, (auc, name) in enumerate(zip(model_aucs, model_names)):
        valid_mask = ~np.isnan(auc)
        if np.any(valid_mask):
            ax2.plot(time_steps[valid_mask], auc[valid_mask], label=name, alpha=0.7, linewidth=1.5)
    valid_ensemble_auc = ~np.isnan(ensemble_auc)
    if np.any(valid_ensemble_auc):
        ax2.plot(time_steps[valid_ensemble_auc], ensemble_auc[valid_ensemble_auc], 
                label="Ensemble", linewidth=2, linestyle='--', color='black')
    ax2.set_xlabel("Number of examples seen", fontsize=11)
    ax2.set_ylabel("Cumulative AUC", fontsize=11)
    ax2.set_title("AUC Over Time", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
