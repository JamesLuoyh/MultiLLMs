#!/usr/bin/env python3
"""
Simple runner for multi-LLM ensembles on multiple-choice QA (MedQA / MMLU).

This script demonstrates how to:
  1. Load a Dataset using lm_polygraph's Dataset abstraction.
  2. Load multiple whitebox models.
  3. Collect per-option logits for each model and cache them.
  4. Run an online ensemble with weighted log pooling and a wager update rule.
  5. Evaluate accuracy and calibration (ECE, ROCAUC, PRAUC, PredictionRejectionArea)
     using existing metric implementations.
  6. Save and optionally plot the per-LLM wager trajectories over time.
"""

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Ensure the local src/ tree (which contains lm_polygraph) is importable without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.utils.multi_llm_ensemble import (
    plot_accuracy_and_auc_over_time,
    LogitCache,
    LogitCacheKey,
    collect_option_logits_for_model,
    run_online_ensemble,
    filter_dataset_by_token_length,
)
from lm_polygraph.ue_metrics import ECE, ROCAUC, PRAUC, PredictionRejectionArea

log = logging.getLogger("lm_polygraph")

# Mapping from simplified model paths to actual HuggingFace model paths
MODEL_PATH_MAP: Dict[str, str] = {
    "aloe-beta-8b": "HPAI-BSC/Llama3.1-Aloe-Beta-8B",
    "aloe-alpha-8b": "HPAI-BSC/Llama3-Aloe-8B-Alpha",
    "llama3-aloe-alpha-8b": "HPAI-BSC/Llama3-Aloe-8B-Alpha",  # Alias
    "math-code-llama-3.1-8b": "EpistemeAI/Math-Code-Llama3.1-8B",
    "math-code-llama": "EpistemeAI/Math-Code-Llama3.1-8B",  # Alias
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "biomistral-7b": "BioMistral/BioMistral-7B",
    "biomistral": "BioMistral/BioMistral-7B",  # Alias for convenience
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # Alias for convenience
    "gemma-7b-it": "google/gemma-7b-it",
    "gemma-7b": "google/gemma-7b-it",  # Alias for convenience
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemma-2-9b": "google/gemma-2-9b-it",  # Alias for convenience
    "financial-advice-llm": "db5kb/financial-advice-llm-Llama-3.1-8B-Instruct",
    "financial-advice-llama-3.1-8b": "db5kb/financial-advice-llm-Llama-3.1-8B-Instruct",  # Alias
    "mistral3-8b-reasoning-2512": "mistralai/Ministral-3-8B-Reasoning-2512",
    "mistral3-8b-reasoning": "mistralai/Ministral-3-8B-Reasoning-2512",  # Alias
    # Add more mappings as needed
}


def resolve_model_paths(model_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Resolve simplified model paths to actual HuggingFace paths.
    
    Args:
        model_paths: List of model paths (simplified or full HF paths)
    
    Returns:
        Tuple of (resolved_paths, simplified_to_actual_map)
    """
    resolved_paths = []
    simplified_to_actual = {}
    
    for path in model_paths:
        # Normalize the path for lookup: lowercase, replace underscores with hyphens
        normalized_key = path.lower().replace("_", "-")
        
        # Try direct match first
        if normalized_key in MODEL_PATH_MAP:
            actual_path = MODEL_PATH_MAP[normalized_key]
            resolved_paths.append(actual_path)
            simplified_to_actual[path] = actual_path
            log.info(f"Resolved simplified path '{path}' -> '{actual_path}'")
        else:
            # Check if it's a path with org prefix (e.g., "HPAI-BSC/aloe-beta-8b")
            # Extract just the model name part
            if "/" in path:
                org, model_name = path.split("/", 1)
                model_key = model_name.lower().replace("_", "-")
                if model_key in MODEL_PATH_MAP:
                    actual_path = MODEL_PATH_MAP[model_key]
                    resolved_paths.append(actual_path)
                    simplified_to_actual[path] = actual_path
                    log.info(f"Resolved path with org prefix '{path}' -> '{actual_path}'")
                else:
                    # Assume it's already a full HF path
                    resolved_paths.append(path)
                    simplified_to_actual[path] = path
                    log.info(f"Using full path '{path}' as-is")
            else:
                # No slash, assume it's a simplified name that's not in our map
                # or it's already a full path without org prefix (unlikely but possible)
                resolved_paths.append(path)
                simplified_to_actual[path] = path
                log.warning(
                    f"Path '{path}' not found in MODEL_PATH_MAP. "
                    f"Using as-is. Available keys: {list(MODEL_PATH_MAP.keys())}"
                )
    
    return resolved_paths, simplified_to_actual


def sanitize_filename(text: str) -> str:
    """
    Sanitize a string to be safe for use in filenames.
    Replaces special characters with underscores.
    """
    # Replace slashes, colons, and other special chars with underscores
    sanitized = re.sub(r'[^\w\-.]', '_', text)
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def generate_output_filename(
    base_name: str,
    dataset_name: str,
    simplified_model_paths: List[str],
    extension: str = "png",
) -> str:
    """
    Generate a distinct filename for output files.
    
    Args:
        base_name: Base name for the file (e.g., "wagers_over_time")
        dataset_name: Dataset name (e.g., "mmlu", "mcq")
        simplified_model_paths: List of simplified model path identifiers
        extension: File extension (default: "png")
    
    Returns:
        Filename string
    """
    # Create a short identifier from model paths
    model_ids = [sanitize_filename(path.lower()) for path in simplified_model_paths]
    model_suffix = "_".join(model_ids)
    
    # Combine components
    filename = f"{base_name}_{dataset_name}_{model_suffix}.{extension}"
    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-LLM ensemble on MCQ datasets.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=["mcq", "mmlu", "gsm8k-mc", "arc-easy", "arc-challenge", "medmcqa", "yesno", "pubmedqa"],
        required=True,
        help=(
            "High-level dataset name. "
            "'mcq' = MedQA (GBaker/MedQA-USMLE-4-options), "
            "'mmlu' = MMLU simple_instruct (['LM-Polygraph/mmlu', 'simple_instruct']), "
            "'gsm8k-mc' = GSM8K Multiple Choice (guipenedo/gsm8k-mc), "
            "'arc-easy' = ARC-Easy (allenai/ai2_arc, ARC-Easy subset), "
            "'arc-challenge' = ARC-Challenge (allenai/ai2_arc, ARC-Challenge subset), "
            "'medmcqa' = MedMCQA (openlifescienceai/medmcqa), "
            "'yesno' = Yes/No questions (requires --hf-dataset-path). "
            "For 'yesno', ensure your dataset has 'question' and 'answer' columns, "
            "where 'answer' is 'yes'/'no', True/False, or 0/1. "
            "'pubmedqa' = PubMedQA (qiaojin/PubMedQA), a medical yes/no question dataset."
        ),
    )
    parser.add_argument(
        "--hf-dataset-path",
        type=str,
        default=None,
        help=(
            "Custom HuggingFace dataset path (e.g., 'username/dataset_name' or ['username/dataset_name', 'config']). "
            "Required for 'yesno' dataset type. Overrides the default dataset for other types if provided."
        ),
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Column name for question text. Defaults based on dataset type.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Column name for answer labels. Defaults based on dataset type.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (e.g., 'test', 'validation').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Batch size for dataset iteration per model. "
            "With parallel processing on multiple GPUs, consider increasing this "
            "to better utilize each GPU's memory and throughput."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of evaluation examples to use (None = use full split).",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more model paths for whitebox models. "
            "Can use simplified names (e.g., 'aloe-beta-8b', 'llama-3.2-3b-instruct') "
            "or full HuggingFace paths (e.g., 'HPAI-BSC/Llama3.1-Aloe-Beta-8B'). "
            "Simplified names will be resolved to full HF paths automatically."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./workdir/output/multi_llm_logits",
        help="Directory to cache per-model logits.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Number of tokens to generate for the answer (MedQA: 1 for single-letter response).",
    )
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Answer option tokens (must each map to a single tokenizer token). "
            "Default: ['A', 'B', 'C', 'D'] for MCQ datasets, ['Yes', 'No'] for yes/no datasets. "
            "You can override this for any dataset type."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the dataset before processing (default: True).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed for dataset shuffling (default: 0).",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable dataset shuffling.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1200,
        help="Maximum number of tokens per prompt. Samples exceeding this will be dropped (default: 1200).",
    )
    return parser.parse_args()


def load_whitebox_model(model_path: str, device_map: Optional[str] = None) -> WhiteboxModel:
    """
    Convenience loader for a whitebox model via WhiteboxModel.from_pretrained.

    For production use, prefer the hydra-based loaders in scripts/polygraph_eval.
    
    Args:
        model_path: Path to the model
        device_map: Device mapping for model (e.g., "cuda:0", "cuda:1", "auto")
    """
    import torch
    
    log.info(f"Loading whitebox model from {model_path} on device {device_map}")
    load_kwargs = {}
    
    # For single GPU assignments, avoid device_map to prevent meta tensor issues
    # Load on CPU first, then move to device
    use_device_map = False
    target_device = None
    
    if device_map is not None:
        if device_map == "auto" or (isinstance(device_map, str) and "," in device_map):
            # Multi-GPU or auto - use device_map
            use_device_map = True
            load_kwargs["device_map"] = device_map
            load_kwargs["low_cpu_mem_usage"] = True
        elif device_map.startswith("cuda:"):
            # Single GPU - extract device and load on CPU, then move
            target_device = torch.device(device_map)
            log.info(f"Single GPU detected, will load on CPU then move to {target_device}")
        else:
            # Unknown format, try device_map anyway
            use_device_map = True
            load_kwargs["device_map"] = device_map
            load_kwargs["low_cpu_mem_usage"] = True
    
    model = WhiteboxModel.from_pretrained(
        model_path=model_path,
        generation_params={
            # Align with typical eval-style decoding: greedy, single token for single-letter response.
            "max_new_tokens": 1,
            "do_sample": False,
            "temperature": 0.0,
            "stop_strings": ["\n"],
        },
        add_bos_token=True,
        instruct=True,
        **load_kwargs,
    )
    
    # Move to target device if we loaded on CPU
    if target_device is not None:
        log.info(f"Moving model to {target_device}")
        model.model = model.model.to(target_device)
    
    return model


def load_dataset(
    dataset_name: str,
    split: str,
    batch_size: int,
    size: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 0,
    hf_dataset_path: Optional[str] = None,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Dataset:
    """
    Load a multiple-choice dataset using Dataset.load.

    For MedQA:
        dataset_name: 'GBaker/MedQA-USMLE-4-options'
        text_column:  'question'
        label_column: 'answer_idx'

    For MMLU (simple_instruct pre-processed variant):
        dataset_name: ['LM-Polygraph/mmlu', 'simple_instruct']
        text_column:  'input'
        label_column: 'output'
    
    For GSM8K-MC (multiple-choice version):
        dataset_name: 'guipenedo/gsm8k-mc'
        text_column:  'question'
        label_column: 'answer_idx' or 'answer'
    
    For ARC-Easy:
        dataset_name: ['allenai/ai2_arc', 'ARC-Easy']
        text_column:  'question'
        label_column: 'answerKey'
        Note: Only questions with answerKey in ['A', 'B', 'C', 'D'] are included.
    
    For ARC-Challenge:
        dataset_name: ['allenai/ai2_arc', 'ARC-Challenge']
        text_column:  'question'
        label_column: 'answerKey'
        Note: Only questions with answerKey in ['A', 'B', 'C', 'D'] are included.
    
    For MedMCQA:
        dataset_name: 'openlifescienceai/medmcqa'
        text_column:  'question'
        label_column: 'answer_idx' or 'cop'
        Note: Uses options dict with keys 'opa', 'opb', 'opc', 'opd' or similar format.
    
    For PubMedQA:
        dataset_name: 'pubmedqa'
        hf_dataset: 'qiaojin/PubMedQA'
        text_column: 'question'
        label_column: 'final_decision' (yes/no/maybe)
        Note: Automatically filters out 'maybe' answers and converts yes/no to 0/1 indices.
    """
    log.info(f"Loading dataset {dataset_name} split={split} size={size}")
    
    # Override with custom HF dataset path if provided
    if hf_dataset_path is not None and dataset_name != "yesno":
        log.info(f"Using custom HF dataset path: {hf_dataset_path}")

    # Map high-level dataset selector to the concrete HF dataset + columns.
    # We only support multiple-choice style datasets here.
    if hf_dataset_path is not None and dataset_name != "yesno":
        # Use custom dataset path, but keep the prompt format from dataset_name
        hf_dataset = hf_dataset_path
        # Use provided columns or defaults
        text_column = text_column or ("question" if dataset_name in ["mcq", "gsm8k-mc", "arc-easy", "arc-challenge", "medmcqa"] else "input")
        label_column = label_column or ("answer_idx" if dataset_name in ["mcq"] else "output" if dataset_name == "mmlu" else "answerKey" if dataset_name in ["arc-easy", "arc-challenge"] else "answer")
        # Use default prompt for the dataset type
        if dataset_name == "mcq" or dataset_name in ["arc-easy", "arc-challenge"]:
            prompt = (
                "Q:{question}\n"
                "A. {option_a}\n"
                "B. {option_b}\n"
                "C. {option_c}\n"
                "D. {option_d}\n"
                "Answer with only the letter (A, B, C, or D):"
            )
            description = "The following are multiple choice questions (with answers). Answer with only the letter (A, B, C, or D), nothing else."
        else:
            prompt = ""
            description = ""
    elif dataset_name == "mcq":
        # MedQA: GBaker/MedQA-USMLE-4-options
        hf_dataset = "GBaker/MedQA-USMLE-4-options"
        text_column = "question"
        label_column = "answer_idx"
        # Mirror the prompt/description used in the default MedQA configs so that
        # single-model performance matches standard MedQA benchmarks.
        # Updated to force single-letter response for better accuracy alignment.
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
    elif dataset_name == "mmlu":
        # MMLU: simple_instruct pre-processed variant
        hf_dataset = ["LM-Polygraph/mmlu", "simple_instruct"]
        text_column = "input"
        label_column = "output"
        # Let Dataset.load handle formatting; we just pass through.
        prompt = ""
        description = ""
    elif dataset_name == "gsm8k-mc":
        # GSM8K Multiple Choice: guipenedo/gsm8k-mc
        # Structure: Question, A, B, C, D, Answer columns
        hf_dataset = "guipenedo/gsm8k-mc"
        text_column = "Question"  # Capital Q as per actual dataset
        label_column = "Answer"  # Capital A as per actual dataset
        # Format similar to MedQA for multiple-choice questions
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
            "about mathematics. Answer with only the letter (A, B, C, or D), "
            "nothing else."
        )
    elif dataset_name == "arc-easy":
        # ARC-Easy: allenai/ai2_arc with ARC-Easy config
        hf_dataset = ["allenai/ai2_arc", "ARC-Easy"]
        text_column = "question"
        label_column = "answerKey"
        # Format similar to MedQA for multiple-choice questions
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
            "about science. Answer with only the letter (A, B, C, or D), "
            "nothing else."
        )
    elif dataset_name == "arc-challenge":
        # ARC-Challenge: allenai/ai2_arc with ARC-Challenge config
        hf_dataset = ["allenai/ai2_arc", "ARC-Challenge"]
        text_column = "question"
        label_column = "answerKey"
        # Format similar to MedQA for multiple-choice questions
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
            "about science. Answer with only the letter (A, B, C, or D), "
            "nothing else."
        )
    elif dataset_name == "medmcqa":
        # MedMCQA: openlifescienceai/medmcqa
        hf_dataset = "openlifescienceai/medmcqa"
        text_column = "question"
        # MedMCQA uses 'cop' (correct option) which is 0, 1, 2, 3 for A, B, C, D
        # or 'answer_idx' if available. We'll need to handle the conversion.
        label_column = "cop"  # Will convert 0,1,2,3 to A,B,C,D
        # Format similar to MedQA for multiple-choice questions
        # MedMCQA uses 'opa', 'opb', 'opc', 'opd' for options
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
    elif dataset_name == "yesno":
        # Generic yes/no dataset format
        # Expected columns: 'question' (text), 'answer' (yes/no/True/False/0/1)
        if hf_dataset_path is None:
            raise ValueError(
                "For yes/no datasets, --hf-dataset-path is required. "
                "Example: --hf-dataset-path 'username/dataset_name'"
            )
        hf_dataset = hf_dataset_path
        text_column = text_column or "question"
        label_column = label_column or "answer"
        # Yes/No prompt template - simple format
        prompt = (
            "Q: {question}\n"
            "Answer with only Yes or No:"
        )
        description = (
            "The following are yes/no questions. "
            "Answer with only 'Yes' or 'No', nothing else."
        )
    elif dataset_name == "pubmedqa":
        # PubMedQA: qiaojin/PubMedQA
        # Structure: 'question' (research question), 'final_decision' (yes/no/maybe)
        hf_dataset = "qiaojin/PubMedQA"
        text_column = text_column or "question"
        label_column = label_column or "final_decision"
        # PubMedQA prompt - includes question and context (if available)
        # PubMedQA questions are medical research questions that can be answered yes/no/maybe
        # We'll format to ask for yes/no only and filter out 'maybe' answers
        prompt = (
            "Q: {question}\n"
            "Answer with only Yes or No:"
        )
        description = (
            "The following are medical research questions that can be answered with yes or no. "
            "Answer with only 'Yes' or 'No', nothing else."
        )
    else:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}'. "
            "Supported datasets: 'mcq' (MedQA), 'mmlu' (MMLU simple_instruct), "
            "'gsm8k-mc' (GSM8K Multiple Choice), 'arc-easy' (ARC-Easy), "
            "'arc-challenge' (ARC-Challenge), 'medmcqa' (MedMCQA), 'yesno' (Yes/No questions), "
            "'pubmedqa' (PubMedQA)."
        )
    # Parse hf_dataset if it's a string representation of a list
    if isinstance(hf_dataset, str) and hf_dataset.startswith('[') and hf_dataset.endswith(']'):
        import ast
        try:
            hf_dataset = ast.literal_eval(hf_dataset)
        except:
            pass
    
    ds = Dataset.load(
        hf_dataset,
        text_column,
        label_column,
        batch_size=batch_size,
        prompt=prompt,
        description=description,
        split=split,
        size=size,
        load_from_disk=False,
    )
    log.info(f"Loaded dataset with {len(ds.x)} examples")
    
    # For yes/no datasets (including pubmedqa), convert labels to indices (0 for Yes, 1 for No)
    if dataset_name in ["yesno", "pubmedqa"]:
        log.info(f"Converting yes/no labels to indices for {dataset_name} (Yes=0, No=1)")
        converted_labels = []
        filtered_x = []
        filtered_indices = []
        
        for idx, (label, text) in enumerate(zip(ds.y, ds.x)):
            # Handle various formats: "yes"/"no", True/False, 0/1, "Yes"/"No"
            label_str = str(label).lower().strip()
            
            # For PubMedQA, filter out "maybe" answers
            if dataset_name == "pubmedqa" and label_str == "maybe":
                log.debug(f"Skipping example {idx} with 'maybe' label")
                continue
            
            if label_str in ["yes", "true", "1"]:
                converted_labels.append(0)  # Yes maps to index 0
                filtered_x.append(text)
                filtered_indices.append(idx)
            elif label_str in ["no", "false", "0"]:
                converted_labels.append(1)  # No maps to index 1
                filtered_x.append(text)
                filtered_indices.append(idx)
            else:
                # For PubMedQA, skip unknown labels; for yesno, warn and assume yes
                if dataset_name == "pubmedqa":
                    log.warning(f"Skipping example {idx} with unexpected label format: {label}")
                    continue
                else:
                    log.warning(f"Unexpected label format: {label}. Assuming 'yes'.")
                    converted_labels.append(0)
                    filtered_x.append(text)
                    filtered_indices.append(idx)
        
        # Update dataset with filtered data
        if filtered_x:
            ds.x = filtered_x
            ds.y = converted_labels
            if hasattr(ds, 'images') and ds.images is not None:
                ds.images = [ds.images[i] for i in filtered_indices]
            log.info(
                f"Label conversion complete. Filtered {len(ds.x)} examples. "
                f"Yes count: {sum(1 for l in ds.y if l == 0)}, "
                f"No count: {sum(1 for l in ds.y if l == 1)}"
            )
        else:
            raise ValueError(
                f"No valid examples found after filtering {dataset_name}. "
                "Ensure the dataset has 'yes'/'no' labels (and for PubMedQA, excludes 'maybe')."
            )
    
    # Shuffle dataset if requested
    if shuffle:
        log.info(f"Shuffling dataset with seed={shuffle_seed}")
        rng = np.random.RandomState(shuffle_seed)
        indices = np.arange(len(ds.x))
        rng.shuffle(indices)
        ds.select(indices.tolist())
        log.info("Dataset shuffled")
    
    return ds


def process_single_model(
    model_idx: int,
    model_path: str,
    device_map: Optional[str],
    dataset: Dataset,
    cache: LogitCache,
    dataset_name: str,
    split: str,
    options: List[str],
    max_new_tokens: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Process a single model to collect logits. This function is designed to be called
    in parallel for multiple models.
    
    Returns:
        Tuple of (model_idx, logits, labels)
    """
    log.info(f"[Model {model_idx}] Starting processing for {model_path} on device {device_map}")
    key = LogitCacheKey(
        dataset_name=dataset_name,
        split=split,
        model_id=model_path,
        num_examples=len(dataset.x),
    )
    cache_path = cache.path_for(key)
    
    if cache_path.exists():
        log.info(f"[Model {model_idx}] Found cached logits at {cache_path}, loading...")
        logits_i, labels_i = cache.load(key)
    else:
        log.info(f"[Model {model_idx}] Loading model {model_path}...")
        model = load_whitebox_model(model_path, device_map=device_map)
        log.info(f"[Model {model_idx}] Collecting logits for {model_path}...")
        logits_i, labels_i = collect_option_logits_for_model(
            model=model,
            dataset=dataset,
            option_tokens=options,
            max_new_tokens=max_new_tokens,
        )
        log.info(f"[Model {model_idx}] Saving logits to cache...")
        cache.save(key, logits_i, labels_i, meta={"options": options})
        log.info(f"[Model {model_idx}] Completed processing for {model_path}")
    
    return (model_idx, logits_i, labels_i)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # Resolve simplified model paths to actual HF paths
    log.info(f"Input model paths: {args.model_paths}")
    resolved_model_paths, simplified_to_actual_map = resolve_model_paths(args.model_paths)
    log.info(f"Resolved model paths: {resolved_model_paths}")
    
    # Store the mapping for reference
    log.info(f"Model path mapping: {json.dumps(simplified_to_actual_map, indent=2)}")

    # Detect available GPUs
    try:
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        log.info(f"Detected {num_gpus} GPU(s) available")
    except ImportError:
        num_gpus = 0
        log.warning("PyTorch not available, cannot detect GPUs. Assuming CPU-only mode.")
    
    # Assign GPUs to models (round-robin if more models than GPUs)
    num_models = len(resolved_model_paths)
    device_assignments: List[Optional[str]] = []
    if num_gpus > 0:
        for i in range(num_models):
            gpu_id = i % num_gpus
            device_assignments.append(f"cuda:{gpu_id}")
        log.info(f"GPU assignments: {dict(zip(resolved_model_paths, device_assignments))}")
    else:
        device_assignments = [None] * num_models
        log.info("No GPUs detected, models will use default device (CPU or auto)")

    cache = LogitCache(args.cache_dir)

    # Determine default options if not provided
    if args.options is None:
        if args.dataset_name in ["yesno", "pubmedqa"]:
            args.options = ["Yes", "No"]
        else:
            args.options = ["A", "B", "C", "D"]
    log.info(f"Using options: {args.options}")
    
    # Load dataset once
    dataset = load_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        batch_size=args.batch_size,
        size=args.num_examples,
        shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed,
        hf_dataset_path=args.hf_dataset_path,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    
    # Filter dataset by token length
    # We need to load the first model first to get its tokenizer for filtering
    # Load on CPU temporarily just for tokenizer (will be moved to GPU later if parallel)
    if args.max_prompt_tokens > 0:
        log.info(f"Loading first model to get tokenizer for filtering (model: {resolved_model_paths[0]})")
        # Load on CPU temporarily to get tokenizer, then we'll reload on GPU if needed
        temp_model = load_whitebox_model(resolved_model_paths[0], device_map="cpu")
        tokenizer = getattr(temp_model, "tokenizer", None)
        dataset = filter_dataset_by_token_length(
            dataset=dataset,
            max_prompt_tokens=args.max_prompt_tokens,
            tokenizer=tokenizer,
        )
        log.info(f"Dataset filtered: {len(dataset.x)} examples remaining")
        # Clean up temp model to free memory
        del temp_model
        import gc
        gc.collect()

    # Collect or load logits for each model in parallel
    # Limit max_workers to num_gpus to avoid multiple models competing on same GPU
    # If num_gpus is 0, fall back to sequential processing (max_workers=1)
    max_workers = max(1, min(num_models, num_gpus) if num_gpus > 0 else 1)
    if num_gpus > 0 and num_models > num_gpus:
        log.warning(
            f"Warning: {num_models} models but only {num_gpus} GPU(s). "
            f"Models will share GPUs (round-robin assignment). "
            f"Consider using fewer models or more GPUs to avoid GPU memory issues."
        )
    log.info(f"Processing {num_models} model(s) with up to {max_workers} workers on {num_gpus} GPU(s)")
    all_model_logits: List[np.ndarray] = [None] * num_models  # type: ignore
    labels_ref: np.ndarray = None  # type: ignore
    
    # Use ThreadPoolExecutor for parallel processing
    # ThreadPoolExecutor works well here because PyTorch models on different GPUs
    # can run concurrently without GIL issues
    # Limit workers to number of GPUs to avoid GPU memory competition
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {}
        for model_idx, (model_path, device_map) in enumerate(zip(resolved_model_paths, device_assignments)):
            future = executor.submit(
                process_single_model,
                model_idx=model_idx,
                model_path=model_path,
                device_map=device_map,
                dataset=dataset,
                cache=cache,
                dataset_name=str(args.dataset_name),
                split=args.split,
                options=args.options,
                max_new_tokens=args.max_new_tokens,
            )
            future_to_idx[future] = model_idx
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                model_idx, logits_i, labels_i = future.result()
                all_model_logits[model_idx] = logits_i
                
                if labels_ref is None:
                    labels_ref = labels_i
                else:
                    if not np.array_equal(labels_ref, labels_i):
                        raise ValueError(
                            f"Label mismatch between models; ensure identical dataset ordering."
                        )
                log.info(f"[Model {model_idx}] Successfully completed processing")
            except Exception as e:
                model_idx = future_to_idx[future]
                log.error(f"[Model {model_idx}] Error during processing: {e}", exc_info=True)
                raise
    
    # Verify all models were processed
    if any(logits is None for logits in all_model_logits):
        raise RuntimeError("Some models failed to process. Check logs above for errors.")

    if labels_ref is None:
        raise RuntimeError("No logits collected; check model_paths and dataset settings.")

    # ------------------------------------------------------------------
    # Per-model standalone metrics (useful for debugging / ablations).
    # We treat each model independently, converting its logits to
    # probabilities and then computing accuracy and UE metrics.
    # ------------------------------------------------------------------
    for model_path, logits_i in zip(resolved_model_paths, all_model_logits):
        # Convert logits to probabilities with a numerically stable softmax
        logits_i = logits_i.astype(np.float32)
        max_i = np.max(logits_i, axis=1, keepdims=True)
        stabilized_i = logits_i - max_i
        log_z_i = max_i + np.log(np.exp(stabilized_i).sum(axis=1, keepdims=True))
        probs_i = np.exp(logits_i - log_z_i)  # [N, K]

        pred_i = probs_i.argmax(axis=1)
        accuracy_i = float(np.mean(pred_i == labels_ref))
        log.info(f"[Per-model] {model_path} accuracy: {accuracy_i:.4f}")

        max_probs_i = probs_i.max(axis=1)
        estimator_i = 1.0 - max_probs_i  # higher = more uncertain
        target_binary_i = (pred_i == labels_ref).astype(int)

        ece_metric_i = ECE(normalize=True)
        ece_value_i = ece_metric_i(
            estimator=estimator_i.tolist(),
            target=target_binary_i.tolist(),
        )
        log.info(
            f"[Per-model] {model_path} ECE (normalize=True): {ece_value_i:.4f}"
        )

        roc_metric_i = ROCAUC()
        # For ROCAUC, use confidence (max_probs) directly, not uncertainty
        # ROCAUC expects higher scores to correspond to positive class (correct=1)
        roc_value_i = roc_metric_i(
            estimator=max_probs_i.tolist(),  # Use confidence, not uncertainty
            target=target_binary_i.tolist(),
        )
        log.info(
            f"[Per-model] {model_path} ROCAUC (uncertainty vs correctness): {roc_value_i:.4f}"
        )

        pr_metric_i = PRAUC(positive_class=1, negative_class=0)
        pr_value_i = pr_metric_i(
            estimator=estimator_i.tolist(),
            target=target_binary_i.tolist(),
        )
        log.info(
            f"[Per-model] {model_path} PRAUC (uncertainty vs correctness): {pr_value_i:.4f}"
        )

        prr_metric_i = PredictionRejectionArea(max_rejection=1.0)
        prr_value_i = prr_metric_i(
            estimator=estimator_i.tolist(),
            target=target_binary_i.astype(float).tolist(),
        )
        log.info(
            f"[Per-model] {model_path} Prediction-Rejection Area: {prr_value_i:.4f}"
        )

    # Run online ensemble with log pooling and wager update
    ensemble_result = run_online_ensemble(
        all_model_logits=all_model_logits,
        labels=labels_ref,
        initial_wagers=None,
    )
    pooled_probs = ensemble_result["pooled_probs"]  # [N, K]
    pooled_pred = ensemble_result["pooled_pred"]  # [N]
    labels = ensemble_result["labels"]  # [N]
    wagers_history = ensemble_result.get("wagers_history")  # [N+1, L]

    # Compute accuracy
    accuracy = float(np.mean(pooled_pred == labels))
    log.info(f"Ensemble accuracy: {accuracy:.4f}")

    # Prepare confidence (1 - uncertainty) style scores for UE metrics
    # We use negative max-prob as "uncertainty" so that higher = more uncertain
    max_probs = pooled_probs.max(axis=1)  # [N]
    # For ECE / ROCAUC / PRAUC we follow the convention that estimator is "uncertainty".
    estimator = 1.0 - max_probs  # higher = more uncertain
    target_binary = (pooled_pred == labels).astype(int)  # correctness as 0/1

    # ECE
    ece_metric = ECE(normalize=True)
    ece_value = ece_metric(estimator=estimator.tolist(), target=target_binary.tolist())
    log.info(f"Ensemble ECE (normalize=True): {ece_value:.4f}")

    # ROCAUC (treating correctness as binary label)
    # For ROCAUC, use confidence (max_probs) directly, not uncertainty
    # ROCAUC expects higher scores to correspond to positive class (correct=1)
    roc_metric = ROCAUC()
    roc_value = roc_metric(estimator=max_probs.tolist(), target=target_binary.tolist())  # Use confidence, not uncertainty
    log.info(f"Ensemble ROCAUC (uncertainty vs correctness): {roc_value:.4f}")

    # PRAUC
    pr_metric = PRAUC(positive_class=1, negative_class=0)
    pr_value = pr_metric(estimator=estimator.tolist(), target=target_binary.tolist())
    log.info(f"Ensemble PRAUC (uncertainty vs correctness): {pr_value:.4f}")

    # Prediction-Rejection Area (full curve)
    prr_metric = PredictionRejectionArea(max_rejection=1.0)
    prr_value = prr_metric(
        estimator=estimator.tolist(), target=target_binary.astype(float).tolist()
    )
    log.info(f"Ensemble Prediction-Rejection Area: {prr_value:.4f}")

    # Set up cache directory for saving outputs
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the model path mapping for reference
    mapping_path = cache_dir / "model_path_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(simplified_to_actual_map, f, indent=2)
    log.info(f"Saved model path mapping to {mapping_path}")

    # Save wagers history and, if matplotlib is available, plot trajectories
    if wagers_history is not None:
        # Generate distinct filename using simplified paths
        wagers_base_filename = generate_output_filename(
            "wagers_over_time",
            args.dataset_name,
            args.model_paths,  # Use original (simplified) paths for filename
            extension="npy",
        )
        wagers_path = cache_dir / wagers_base_filename
        np.save(wagers_path, wagers_history)
        log.info(f"Saved wagers history to {wagers_path}")

        try:
            import matplotlib.pyplot as plt

            steps = np.arange(wagers_history.shape[0])
            for i, model_path in enumerate(resolved_model_paths):
                # Use simplified path for label if available, otherwise use model name
                simplified_path = args.model_paths[i]
                label = Path(model_path).name if simplified_path == model_path else simplified_path
                plt.plot(steps, wagers_history[:, i], label=label)

            plt.xlabel("Question index")
            plt.ylabel("Wager")
            plt.title("Per-LLM wager trajectory over questions")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = cache_dir / wagers_base_filename.replace(".npy", ".png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.info(f"Saved wager trajectory plot to {plot_path}")
        except Exception as e:
            log.warning(f"Could not generate wager plot: {e}")

    # Generate accuracy and AUC over time plots
    try:
        # Use simplified paths for model names in the plot
        model_names = []
        for i, resolved_path in enumerate(resolved_model_paths):
            simplified_path = args.model_paths[i]
            # Use simplified path if it's different from resolved, otherwise use model name
            if simplified_path != resolved_path:
                model_names.append(simplified_path)
            else:
                model_names.append(Path(resolved_path).name)
        
        # Generate distinct filename using simplified paths
        accuracy_auc_filename = generate_output_filename(
            "accuracy_auc_over_time",
            args.dataset_name,
            args.model_paths,  # Use original (simplified) paths for filename
        )
        accuracy_auc_plot_path = cache_dir / accuracy_auc_filename
        plot_accuracy_and_auc_over_time(
            all_model_logits=all_model_logits,
            ensemble_result=ensemble_result,
            model_names=model_names,
            save_path=accuracy_auc_plot_path,
        )
        log.info(f"Successfully generated accuracy and AUC plot at {accuracy_auc_plot_path}")
    except Exception as e:
        import traceback
        log.error(f"Could not generate accuracy and AUC plot: {e}")
        log.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()


