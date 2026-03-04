#!/usr/bin/env python3
"""
Pipeline to:
  1. Load a multiple-choice dataset (MMLU or MedQA).
  2. Run a single whitebox LLM (e.g., Llama 3.1 8B Instruct) on the dataset.
  3. Record, for each example:
       - Option logits (A/B/C/D),
       - Ground-truth label index,
       - Last-layer hidden state at the final prompt token.
  4. Train an MLP (Multi-Layer Perceptron) regressor on hidden states to predict the True Class Probability (TCP).
     TCP is the probability assigned to the correct/true class label.
  5. Compare:
       - Model MSE (predicted TCP vs. actual TCP) vs.
       - MLP MSE at predicting TCP.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Ensure the local src/ tree (which contains lm_polygraph) is importable without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.multi_llm_ensemble import _resolve_option_token_ids, filter_dataset_by_token_length

log = logging.getLogger("lm_polygraph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a whitebox LLM on MMLU/MedQA, collect logits and hidden states, "
            "and train an MLP regressor to predict True Class Probability (TCP)."
        )
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mmlu",
        choices=["mmlu", "medqa"],
        help="Dataset to use: 'mmlu' (LM-Polygraph/mmlu, simple_instruct) or 'medqa' (GBaker/MedQA-USMLE-4-options).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split(s) to evaluate on. Can be a single split (e.g., 'test', 'train') or comma-separated (e.g., 'train,test') or 'both' for train+test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for dataset iteration.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of evaluation examples to use (None = use full split).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="HF model path for the whitebox model (e.g., meta-llama/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--option-tokens",
        type=str,
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Answer option tokens (must each map to a single tokenizer token).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of examples for the held-out test set for the MLP regressor.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/test split of the MLP regressor.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of the dataset (shuffling is enabled by default).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed for shuffling the dataset (default: 0).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens per prompt. Samples exceeding this will be dropped (default: 1200).",
    )
    parser.add_argument(
        "--layer-start",
        type=int,
        default=None,
        help="Starting layer index to extract (0-based, where 0 is first transformer layer, -1 is last, -15 is 15th-to-last). Default: None (extract last num_layers).",
    )
    parser.add_argument(
        "--layer-end",
        type=int,
        default=None,
        help="Ending layer index to extract (exclusive, 0-based, where 0 is first transformer layer, -1 is last, -15 is 15th-to-last). Default: None (extract last num_layers).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/common/users/yl2310/MultiLLMs",
        help="Directory to store cached hidden states and tcp (default: /common/users/yl2310/MultiLLMs).",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Specific cache file path (overrides auto-generated cache file name).",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cache file exists.",
    )
    parser.add_argument(
        "--use-max-probs",
        dest="use_max_probs",
        action="store_true",
        default=True,
        help="Include max_probs as a feature when training the predictor (default: True).",
    )
    parser.add_argument(
        "--no-max-probs",
        dest="use_max_probs",
        action="store_false",
        help="Exclude max_probs from predictor features (use only hidden states).",
    )
    return parser.parse_args()


def load_dataset(
    dataset_name: str,
    split: str,
    batch_size: int,
    size: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
) -> Dataset:
    """
    Load a multiple-choice dataset using Dataset.load.
    Supports loading multiple splits and combining them.

    Args:
        dataset_name: Either 'mmlu' or 'medqa'
        split: Dataset split(s) to load
        batch_size: Batch size for dataset iteration
        size: Optional limit on number of examples
        shuffle: Whether to shuffle the dataset
        shuffle_seed: Random seed for shuffling

    Returns:
        Dataset instance with loaded data

    For MMLU (simple_instruct pre-processed variant):
        dataset_name: ['LM-Polygraph/mmlu', 'simple_instruct']
        text_column:  'input'
        label_column: 'output'

    For MedQA:
        dataset_name: 'GBaker/MedQA-USMLE-4-options'
        text_column: 'question'
        label_column: 'answer_idx'
    """
    # Map dataset name to HF dataset path and column names
    if dataset_name == "mmlu":
        hf_dataset = ["LM-Polygraph/mmlu", "simple_instruct"]
        text_column = "input"
        label_column = "output"
        prompt = ""
        description = ""
    elif dataset_name == "medqa":
        hf_dataset = "GBaker/MedQA-USMLE-4-options"
        text_column = "question"
        label_column = "answer_idx"
        # Use the prompt/description from the default MedQA configs
        prompt = (
            "Q:{question}\n"
            "A. {option_a}\n"
            "B. {option_b}\n"
            "C. {option_c}\n"
            "D. {option_d}\n"
            "Answer with only the letter (A, B, C, or D):"
        )
        description = (
            "The following are multiple choice questions (with answers) "
            "about medical knowledge. Answer with only the letter (A, B, C, or D), "
            "nothing else."
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}'. "
            "Supported datasets: 'mmlu', 'medqa'"
        )

    # Handle special case: "both" means train+test
    if split.lower() == "both":
        splits = ["train", "test"]
    elif "," in split:
        splits = [s.strip() for s in split.split(",")]
    else:
        splits = [split]

    log.info(f"Loading {dataset_name} dataset splits={splits} size={size}")
    
    # Load each split and combine them
    datasets = []
    for split_name in splits:
        ds = Dataset.load(
            hf_dataset,
            text_column,
            label_column,
            batch_size=batch_size,
            prompt=prompt,
            description=description,
            split=split_name,
            size=size,
            load_from_disk=False,
        )
        datasets.append(ds)
        log.info(f"Loaded {split_name} split with {len(ds.x)} examples")
    
    # Combine all datasets
    if len(datasets) > 1:
        combined_x = []
        combined_y = []
        combined_images = None if datasets[0].images is None else []
        
        for ds in datasets:
            combined_x.extend(ds.x)
            combined_y.extend(ds.y)
            if combined_images is not None:
                if ds.images is not None:
                    combined_images.extend(ds.images)
                else:
                    combined_images.extend([None] * len(ds.x))
        
        ds = Dataset(combined_x, combined_y, batch_size, images=combined_images)
        log.info(f"Combined {len(splits)} splits into {len(ds.x)} total examples")
    
    if shuffle:
        log.info(f"Shuffling {dataset_name} dataset with seed={shuffle_seed}")
        rng = np.random.default_rng(shuffle_seed)
        indices = np.arange(len(ds.x))
        rng.shuffle(indices)
        ds.x = [ds.x[i] for i in indices]
        ds.y = [ds.y[i] for i in indices]
        if ds.images is not None:
            ds.images = [ds.images[i] for i in indices]
    
    log.info(f"Final {dataset_name} dataset has {len(ds.x)} examples")
    return ds


def load_whitebox_model(model_path: str) -> WhiteboxModel:
    """
    Convenience loader for a whitebox model via WhiteboxModel.from_pretrained.
    """
    import torch
    # Determine device - use CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device_map = "cuda:0"
        log.info(f"Loading whitebox model from {model_path} on GPU (cuda:0)")
    else:
        device_map = "cpu"
        log.warning(f"CUDA not available, loading model on CPU (will be very slow!)")
    
    model = WhiteboxModel.from_pretrained(
        model_path=model_path,
        generation_params={
            # We do not actually generate here, but we keep config consistent.
            "max_new_tokens": 3,
            "do_sample": False,
            "temperature": 0.0,
            "stop_strings": ["\n"],
        },
        add_bos_token=True,
        instruct=True,
        device_map=device_map,
    )
    return model


def collect_logits_and_hidden_states(
    model: WhiteboxModel,
    dataset: Dataset,
    option_tokens: List[str],
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    num_layers: int = 15,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[int], int]:
    """
    For each example in the (multiple-choice) dataset:
        - Run the model in forward mode on the prompt only.
        - Extract hidden states from specified layers at the final prompt token.
        - Extract logits for the specified option tokens at that position.

    Args:
        model: WhiteboxModel instance
        dataset: Dataset to process
        option_tokens: List of option tokens (e.g., ['A', 'B', 'C', 'D'])
        layer_start: Starting layer index (0-based, where 0 is first transformer layer, -1 is last, -15 is 15th-to-last)
        layer_end: Ending layer index (exclusive, 0-based, where 0 is first transformer layer, -1 is last, -15 is 15th-to-last)
        num_layers: Number of last layers to extract if layer_start/layer_end not specified (default: 15)

    Returns:
        logits: float32 array, shape [num_examples, num_options]
        labels: int32 array, shape [num_examples]
        hidden_states_list: List of float32 arrays, each of shape [num_examples, hidden_dim]
            One array per layer (from layer_start to layer_end, or last num_layers if not specified)
        layer_numbers: List of int, the 0-based layer numbers (0 = first transformer layer, -1 = last, etc.)
        num_total_layers: int, total number of transformer layers
    """
    model_device = model.device()
    option_token_ids = _resolve_option_token_ids(model, option_tokens)

    all_logits: List[np.ndarray] = []
    all_labels: List[int] = []
    
    # Will be set on first batch
    num_total_layers = None
    layer_indices = None

    for batch_x, batch_y, _ in dataset:
        # Tokenize prompts
        batch = model.tokenize(batch_x)
        batch = {k: v.to(model_device) for k, v in batch.items()}

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model.model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
            )

        # outputs.logits: [B, T, V], outputs.hidden_states: tuple(layer)[B, T, H]
        logits = outputs.logits[:, -1, :]  # [B, V] at final prompt token
        
        # outputs.hidden_states is a tuple where index 0 is embeddings, 1..N are transformer layers
        hidden_states = outputs.hidden_states
        
        # Calculate layer indices on first batch (they're the same for all batches)
        if num_total_layers is None:
            num_total_layers = len(hidden_states) - 1  # -1 because first is embeddings
            
            # Determine which layers to extract
            if layer_start is not None or layer_end is not None:
                # Convert negative indices to positive
                if layer_start is None:
                    layer_start = 0
                if layer_end is None:
                    layer_end = num_total_layers
                
                # Convert to actual indices (0-based, where 0 = first transformer layer = index 1 in hidden_states)
                # Positive indices: 0, 1, 2, ... -> 1, 2, 3, ... in hidden_states
                # Negative indices: -1, -2, ... -> num_total_layers, num_total_layers-1, ... in hidden_states
                # layer_end is inclusive (so -15 means include layer -15, which is the 15th-to-last layer)
                if layer_start < 0:
                    start_idx = num_total_layers + layer_start + 1  # +1 because we want index in hidden_states
                else:
                    start_idx = layer_start + 1  # +1 to skip embedding layer
                
                if layer_end < 0:
                    # For negative indices, -15 means the 15th-to-last layer (inclusive)
                    # If num_total_layers=32, then -15 means layer 18 (32-15+1=18 in 0-based, which is index 19 in hidden_states)
                    end_idx = num_total_layers + layer_end + 2  # +2: +1 for hidden_states index, +1 to make range inclusive
                else:
                    end_idx = layer_end + 2  # +2: +1 for hidden_states index, +1 to make range inclusive
                
                layer_indices = list(range(start_idx, end_idx))
                log.info(
                    f"Model has {num_total_layers} transformer layers. "
                    f"Extracting layers {layer_start} to {layer_end} (indices {layer_indices} in hidden_states)"
                )
            else:
                # Default: get the last num_layers layers
                layer_start_idx = max(1, num_total_layers - num_layers + 1)
                layer_indices = list(range(layer_start_idx, len(hidden_states)))
                log.info(
                    f"Model has {num_total_layers} transformer layers. "
                    f"Extracting last {num_layers} layers: {layer_indices}"
                )
            
            # Initialize storage for the selected layers
            all_hidden_per_layer: List[List[np.ndarray]] = [[] for _ in range(len(layer_indices))]
        
        # Extract hidden state at final token position for each selected layer
        for i, layer_idx in enumerate(layer_indices):
            hidden_layer = hidden_states[layer_idx][:, -1, :]  # [B, H] at final prompt token
            all_hidden_per_layer[i].append(hidden_layer.cpu().numpy().astype(np.float32))

        # Extract logits for each option token
        batch_option_logits = torch.stack(
            [logits[:, tid] for tid in option_token_ids], dim=-1
        )  # [B, num_options]

        all_logits.append(batch_option_logits.cpu().numpy().astype(np.float32))

        # Convert labels (which may be strings like 'A'/'B'/... or indices) to indices
        for y in batch_y:
            if isinstance(y, str):
                idx = option_tokens.index(y)
            else:
                idx = int(y)
            all_labels.append(idx)

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.asarray(all_labels, dtype=np.int32)
    
    # Concatenate hidden states for each layer
    hidden_states_list = [
        np.concatenate(layer_hidden, axis=0) for layer_hidden in all_hidden_per_layer
    ]
    
    # Convert layer_indices (hidden_states indices) to 0-based layer numbers
    # hidden_states index 1 -> layer 0, index 2 -> layer 1, etc.
    # Also calculate negative indices: layer N -> layer -(num_total_layers - N)
    layer_numbers = []
    for hs_idx in layer_indices:
        layer_0_based = hs_idx - 1  # Convert from hidden_states index to 0-based layer number
        layer_numbers.append(layer_0_based)
    
    # Log which layers were extracted
    if layer_start is not None or layer_end is not None:
        log.info(
            f"Successfully extracted hidden states from layers {layer_start} to {layer_end} "
            f"(0-based layers {layer_numbers}, hidden_states indices {layer_indices} out of {num_total_layers} total transformer layers)"
        )
    else:
        log.info(
            f"Successfully extracted hidden states from last {len(layer_indices)} layers "
            f"(0-based layers {layer_numbers}, hidden_states indices {layer_indices} out of {num_total_layers} total transformer layers)"
        )
    
    return logits_arr, labels_arr, hidden_states_list, layer_numbers, num_total_layers


def get_cache_file_path(
    cache_dir: str,
    dataset_name: str,
    model_path: str,
    num_layers: int | str,
    num_examples: Optional[int],
    cache_file: Optional[str] = None,
) -> Path:
    """Generate cache file path based on parameters."""
    if cache_file:
        return Path(cache_file)
    
    # Create a safe filename from model_path (replace / with _)
    model_safe = model_path.replace("/", "_").replace("\\", "_")
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Include key parameters in filename
    filename = f"hidden_states_tcp_cache_{dataset_name}_{model_safe}_layers{num_layers}"
    if num_examples:
        filename += f"_n{num_examples}"
    filename += ".pkl"
    
    return cache_dir_path / filename


def save_cache(
    cache_path: Path,
    hidden_states_list: List[np.ndarray],
    max_probs: np.ndarray,
    labels: np.ndarray,
    tcp: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    metadata: dict,
    extracted_layer_numbers: List[int],
    num_total_layers: int,
) -> None:
    """Save hidden states, max_probs, labels, tcp, and train/test indices to cache file."""
    cache_data = {
        "hidden_states_list": hidden_states_list,
        "max_probs": max_probs,
        "labels": labels,
        "tcp": tcp,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "metadata": metadata,
        "extracted_layer_numbers": extracted_layer_numbers,
        "num_total_layers": num_total_layers,
    }
    log.info(f"Saving cache to {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    log.info(f"Cache saved successfully ({cache_path.stat().st_size / 1024 / 1024:.2f} MB)")


def load_cache(cache_path: Path) -> Optional[dict]:
    """Load cached data from file. Returns None if file doesn't exist or is invalid."""
    if not cache_path.exists():
        return None
    
    try:
        log.info(f"Loading cache from {cache_path}")
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
        log.info(f"Cache loaded successfully")
        return cache_data
    except Exception as e:
        log.warning(f"Failed to load cache: {e}")
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Determine layer range
    layer_start = args.layer_start
    layer_end = args.layer_end
    if layer_start is not None or layer_end is not None:
        # Calculate number of layers for cache naming
        # We'll determine the actual count after loading the model
        num_layers_for_cache = f"{layer_start}to{layer_end}"
    else:
        num_layers_for_cache = 15
        layer_start = None
        layer_end = None
    
    # Determine cache file path
    cache_path = get_cache_file_path(
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        num_layers=num_layers_for_cache,
        num_examples=args.num_examples,
        cache_file=args.cache_file,
    )

    # Try to load from cache
    cache_data = None
    if not args.force_recompute:
        cache_data = load_cache(cache_path)
        if cache_data is not None:
            # Validate metadata matches current run
            metadata = cache_data.get("metadata", {})
            expected_metadata = {
                "dataset_name": args.dataset_name,
                "model_path": args.model_path,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "test_size": args.test_size,
                "random_seed": args.random_seed,
                "num_examples": args.num_examples,
            }
            if metadata == expected_metadata:
                log.info("Using cached data (hidden states and tcp)")
                hidden_states_list = cache_data["hidden_states_list"]
                max_probs = cache_data["max_probs"]
                labels = cache_data["labels"]
                tcp = cache_data["tcp"]
                idx_train = cache_data["idx_train"]
                idx_test = cache_data["idx_test"]
                
                num_examples = len(labels)
                num_layers_extracted = len(hidden_states_list)
                hidden_dim = hidden_states_list[0].shape[1] if hidden_states_list else 0
                
                # Reconstruct layer numbers from cache
                if "extracted_layer_numbers" in cache_data and "num_total_layers" in cache_data:
                    extracted_layer_numbers = cache_data["extracted_layer_numbers"]
                    num_total_layers = cache_data["num_total_layers"]
                else:
                    # Fallback: old cache format - need to recalculate
                    log.warning("Cache missing layer numbers metadata. Cache may be from old version. Recomputing...")
                    cache_data = None  # Force recomputation
            else:
                log.warning("Cache metadata mismatch, recomputing...")
                cache_data = None

    # If cache not available or invalid, compute everything
    if cache_data is None:
        log.info("Computing hidden states and tcp (this may take a while)...")
        
        # Load dataset and model
        dataset = load_dataset(
            dataset_name=args.dataset_name,
            split=args.split,
            batch_size=args.batch_size,
            size=args.num_examples,
            shuffle=not args.no_shuffle,
            shuffle_seed=args.shuffle_seed,
        )
        model = load_whitebox_model(args.model_path)
        
        # Filter dataset by token length
        if args.max_prompt_tokens > 0:
            tokenizer = getattr(model, "tokenizer", None)
            dataset = filter_dataset_by_token_length(
                dataset=dataset,
                max_prompt_tokens=args.max_prompt_tokens,
                tokenizer=tokenizer,
            )

        # Collect per-example option logits, labels, and hidden states
        if layer_start is not None or layer_end is not None:
            log.info(f"Collecting option logits and hidden states from layers {layer_start} to {layer_end}...")
        else:
            log.info("Collecting option logits and hidden states from last 15 layers...")
        logits, labels, hidden_states_list, extracted_layer_numbers, num_total_layers = collect_logits_and_hidden_states(
            model=model,
            dataset=dataset,
            option_tokens=args.option_tokens,
            layer_start=layer_start,
            layer_end=layer_end,
            num_layers=15,
        )
        num_examples, num_options = logits.shape
        num_layers_extracted = len(hidden_states_list)
        hidden_dim = hidden_states_list[0].shape[1] if hidden_states_list else 0
        log.info(
            f"Collected logits and hidden states for {num_examples} examples, "
            f"{num_options} options, {num_layers_extracted} layers, hidden_dim={hidden_dim}"
        )

        # Compute model predictions and TCP
        logits = logits.astype(np.float32)
        pred = logits.argmax(axis=1)
        
        # Convert logits to probabilities (softmax) - using max_logits for numerical stability
        max_logits_for_softmax = np.max(logits, axis=1, keepdims=True)
        stabilized = logits - max_logits_for_softmax
        log_z = max_logits_for_softmax + np.log(np.exp(stabilized).sum(axis=1, keepdims=True))
        probs = np.exp(logits - log_z)  # [num_examples, num_options]
        
        # Model confidence: max probability of the predicted option
        max_probs = probs.max(axis=1)  # [num_examples] - confidence in the predicted option
        
        # True Class Probability (TCP): probability assigned to the correct/true class label
        # For each example, get the probability of the true label
        tcp = probs[np.arange(num_examples), labels]  # [num_examples] - probability of true class

        # Train/test split (via indices) so we can apply the same split
        # to hidden states, labels, and model confidences
        indices = np.arange(num_examples)
        # For regression, we don't stratify, but we can still use train_test_split
        idx_train, idx_test = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=args.random_seed,
        )
        
        # Save to cache
        metadata = {
            "dataset_name": args.dataset_name,
            "model_path": args.model_path,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "test_size": args.test_size,
            "random_seed": args.random_seed,
            "num_examples": args.num_examples,
        }
        save_cache(
            cache_path=cache_path,
            hidden_states_list=hidden_states_list,
            max_probs=max_probs,
            labels=labels,
            tcp=tcp,
            idx_train=idx_train,
            idx_test=idx_test,
            metadata=metadata,
            extracted_layer_numbers=extracted_layer_numbers,
            num_total_layers=num_total_layers,
        )

    # Apply split to labels (same for all layers)
    tcp_train = tcp[idx_train]
    tcp_test = tcp[idx_test]
    
    # Apply split to max_probs (optionally concatenated with hidden states)
    max_probs_train = max_probs[idx_train].reshape(-1, 1)  # [train_size, 1]
    max_probs_test = max_probs[idx_test].reshape(-1, 1)    # [test_size, 1]

    # Model MSE on the held-out test set, using actual TCP
    # The "predicted" TCP is just the actual TCP (since we're measuring how well we can predict it)
    # But we can also compute a baseline: what if we always predict the mean TCP?
    mean_tcp_train = np.mean(tcp_train)
    baseline_mse = mean_squared_error(tcp_test, np.full_like(tcp_test, mean_tcp_train))
    split_display = args.split if args.split.lower() != "both" else "train+test"
    log.info(f"Baseline MSE (predicting mean TCP) on {args.dataset_name.upper()} ({split_display}) [held-out]: {baseline_mse:.6f}")
    log.info(f"Mean TCP (train): {mean_tcp_train:.4f}, Mean TCP (test): {np.mean(tcp_test):.4f}")

    # Train separate MLP regressor for each layer
    max_probs_str = " + max_probs" if args.use_max_probs else ""
    log.info(f"Training MLP regressors on hidden states{max_probs_str} from {num_layers_extracted} layers...")
    
    # Apply split to hidden states for each layer
    hidden_train_list = [hidden[idx_train] for hidden in hidden_states_list]
    hidden_test_list = [hidden[idx_test] for hidden in hidden_states_list]
    
    # Store results for each layer
    layer_mses = []
    display_layer_numbers = []
    
    # Use the actual extracted layer numbers, converting to negative indexing for display
    # where -1 is the last layer, -2 is second-to-last, etc.
    for layer_idx, layer_0_based in enumerate(extracted_layer_numbers):
        # Convert 0-based to negative indexing: layer N -> -(num_total_layers - N - 1)
        # e.g., if num_total_layers=32: layer 0 -> -32, layer 17 -> -15, layer 31 -> -1
        relative_layer_num = layer_0_based - num_total_layers
        display_layer_numbers.append(relative_layer_num)
        
        # Optionally concatenate hidden states with max_probs for this layer
        if args.use_max_probs:
            X_train_layer = np.concatenate([hidden_train_list[layer_idx], max_probs_train], axis=1)
            X_test_layer = np.concatenate([hidden_test_list[layer_idx], max_probs_test], axis=1)
            feature_desc = f"hidden_dim + 1 for max_probs"
        else:
            X_train_layer = hidden_train_list[layer_idx]
            X_test_layer = hidden_test_list[layer_idx]
            feature_desc = f"hidden_dim only (max_probs excluded)"
        
        log.info(
            f"Training MLP regressor for layer {relative_layer_num} (0-based: {layer_0_based}): "
            f"train_size={X_train_layer.shape[0]}, test_size={X_test_layer.shape[0]}, "
            f"feature_dim={X_train_layer.shape[1]} ({feature_desc})"
        )
        
        # Normalize features for MLP (neural networks typically need normalized inputs)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_layer)
        X_test_scaled = scaler.transform(X_test_layer)
        
        # Train MLP regressor
        reg = MLPRegressor(
            hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        
        reg.fit(X_train_scaled, tcp_train)
        
        # Get TCP predictions for MSE calculation
        tcp_test_pred = reg.predict(X_test_scaled)
        mse_layer = mean_squared_error(tcp_test, tcp_test_pred)
        layer_mses.append(mse_layer)
        
        feature_str = "hidden_state + max_probs" if args.use_max_probs else "hidden_state only"
        log.info(
            f"Layer {relative_layer_num} MLP MSE ({feature_str}) at predicting TCP (held-out): {mse_layer:.6f}"
        )

    # Print summary results
    split_display = args.split if args.split.lower() != "both" else "train+test"
    print("\n" + "="*70)
    print(f"Results for {args.dataset_name.upper()} ({split_display}):")
    print("="*70)
    print(f"Baseline MSE (predicting mean TCP): {baseline_mse:.6f}")
    print(f"Mean TCP (train): {mean_tcp_train:.4f}")
    print(f"Mean TCP (test): {np.mean(tcp_test):.4f}")
    print(f"Std TCP (test): {np.std(tcp_test):.4f}")
    print(f"\nPer-Layer MLP MSE at predicting TCP:")
    print("-"*70)
    for layer_num, mse_val in zip(display_layer_numbers, layer_mses):
        print(f"  Layer {layer_num:3d}: {mse_val:.6f}")
    print("-"*70)
    print(f"Best layer: {display_layer_numbers[np.argmin(layer_mses)]} (MSE: {min(layer_mses):.6f})")
    print(f"Worst layer: {display_layer_numbers[np.argmax(layer_mses)]} (MSE: {max(layer_mses):.6f})")
    print(f"Mean across layers: {np.mean(layer_mses):.6f}")
    print("="*70)


if __name__ == "__main__":
    # Local import to avoid polluting global namespace for tooling
    import torch  # noqa: F401

    main()

