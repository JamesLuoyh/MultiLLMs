#!/usr/bin/env python3
"""
Two-phase training pipeline for distribution-shift experiments.

Key features:
- Explicit phase 1 -> phase 2 transition with guaranteed checkpoint resume.
- Optional per-phase PubMedQA context routing policies.
- Optional model holdout via wagering method config overrides per phase.
- Per-batch metric export and plots:
  - rolling average over last X batches
  - cumulative average since phase start
- Multi-method comparison plots across the same two-phase scenario.

Usage:
  python scripts/wagering_two_phase_pipeline.py <config.yaml>
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from wagering.utils import load_and_merge_configs


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_module = load_module_from_path("wagering_train", SCRIPTS_PATH / "wagering_train.py")
eval_module = load_module_from_path("wagering_eval", SCRIPTS_PATH / "wagering_eval.py")

train_main = train_module.main
eval_main = eval_module.main

log = logging.getLogger("wagering")


DEFAULT_METRICS = [
    "accuracy",
    "ece",
    "kendall_tau",
    "best_model_mrr",
    "brier_d_regret",
]


def _aggregate_eval_metrics(
    eval_results: Optional[Dict[str, Any]],
    metrics: List[str],
) -> Dict[str, Optional[float]]:
    aggregated: Dict[str, Optional[float]] = {m: None for m in metrics}
    if not isinstance(eval_results, dict):
        return aggregated

    for metric in metrics:
        values: List[float] = []
        for _, dataset_result in eval_results.items():
            if not isinstance(dataset_result, dict):
                continue
            value = dataset_result.get(metric)
            if value is None:
                continue
            try:
                f_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(f_value):
                continue
            values.append(f_value)

        if values:
            aggregated[metric] = float(np.mean(values))

    return aggregated


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        loaded = yaml.safe_load(f)
    if isinstance(loaded, dict):
        return loaded
    return {}


def _stable_payload_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_method_cache_fingerprint(
    base_cfg: Dict[str, Any],
    phase1_cfg: Dict[str, Any],
    phase2_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    run_evaluation: bool,
) -> str:
    base_without_methods = copy.deepcopy(base_cfg)
    phase_shift_cfg = base_without_methods.get("phase_shift")
    if isinstance(phase_shift_cfg, dict):
        phase_shift_cfg = copy.deepcopy(phase_shift_cfg)
        phase_shift_cfg.pop("methods", None)
        base_without_methods["phase_shift"] = phase_shift_cfg

    payload = {
        "base_config": base_without_methods,
        "phase1": phase1_cfg,
        "phase2": phase2_cfg,
        "method": method_cfg,
        "run_evaluation": bool(run_evaluation),
    }
    return _stable_payload_hash(payload)


def _load_api_keys_from_config() -> Dict[str, str]:
    """Load API keys from .api_keys.yaml file if it exists."""
    api_keys_path = PROJECT_ROOT / ".api_keys.yaml"
    if not api_keys_path.exists():
        return {}

    with api_keys_path.open("r") as f:
        config = yaml.safe_load(f)
        if not config:
            return {}

        filtered: Dict[str, str] = {}
        for k, v in config.items():
            if v is None or v == "null" or v == "":
                continue
            if isinstance(v, str) and ("your-" in v.lower() and "-here" in v.lower()):
                continue
            filtered[str(k)] = v
        return filtered


def _finish_active_wandb_run() -> None:
    """Best-effort close of any active wandb run in this process."""
    try:
        import wandb
    except Exception:
        return

    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        log.warning("Failed to close active wandb run cleanly: %s", e)


def _latest_transition_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = list(checkpoint_dir.glob("checkpoint_epoch_*_step_*.pt"))
    if not candidates:
        raise RuntimeError(
            f"No resumable checkpoint files found under {checkpoint_dir}. "
            "Expected checkpoint_epoch_*_step_*.pt"
        )
    # Use mtime instead of filename sort: stale files from previous runs can
    # have larger step numbers and be selected incorrectly by lexical ordering.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_phase_config(
    base_cfg: Dict[str, Any],
    phase_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    run_label: str,
    phase_name: str,
    resume_checkpoint: Optional[str] = None,
    device_override: Optional[str] = None,
    emit_resume_checkpoint: bool = False,
    enable_method_wandb: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["wagering_method"] = copy.deepcopy(method_cfg)

    if "datasets" in phase_cfg:
        cfg["datasets"] = copy.deepcopy(phase_cfg["datasets"])

    cfg["num_epochs"] = int(phase_cfg.get("num_epochs", cfg.get("num_epochs", 1)))

    # Validation is not used for early stopping in this pipeline.
    cfg["validation_split_ratio"] = float(phase_cfg.get("validation_split_ratio", 0.0))
    cfg["early_stopping_patience"] = 0

    # Disable implicit auto-resume; phase transition uses explicit checkpoint path.
    cfg["auto_resume"] = False

    # Route PubMedQA prompt policy per phase.
    if "pubmedqa_context_policy" in phase_cfg:
        cfg["pubmedqa_context_policy"] = str(phase_cfg["pubmedqa_context_policy"])

    wm_cfg = cfg.setdefault("wagering_method", {}).setdefault("config", {})

    if device_override is not None:
        wm_cfg["device"] = str(device_override)

    if "frozen_model_indices" in phase_cfg:
        wm_cfg["frozen_model_indices"] = list(phase_cfg.get("frozen_model_indices", []))
    if "inactive_model_indices" in phase_cfg:
        wm_cfg["inactive_model_indices"] = list(phase_cfg.get("inactive_model_indices", []))

    cfg["resume_from_checkpoint"] = resume_checkpoint
    cfg["emit_resume_checkpoint"] = bool(emit_resume_checkpoint)
    cfg["report_to_wandb"] = bool(enable_method_wandb and cfg.get("report_to_wandb", False))

    # Helpful metadata tags for downstream analysis.
    cfg["phase_shift_run_label"] = str(run_label)
    cfg["phase_shift_phase"] = str(phase_name)

    return cfg


def _load_batch_metrics(checkpoint_dir: Path) -> pd.DataFrame:
    path = checkpoint_dir / "batch_metrics.csv"
    if not path.exists():
        raise RuntimeError(
            f"Missing batch metrics file: {path}. "
            "Expected trainer to export batch_metrics.csv"
        )
    df = pd.read_csv(path)
    if "global_step" not in df.columns:
        raise RuntimeError(f"batch_metrics.csv at {path} is missing global_step")
    return df


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _phase_cumulative_mean(series: pd.Series) -> pd.Series:
    return series.expanding(min_periods=1).mean()


def _plot_metric_over_time(
    method_frames: Dict[str, pd.DataFrame],
    metric: str,
    out_path: Path,
    y_label: str,
    title: str,
    value_column: str,
    phase_boundary_column: str,
) -> None:
    plt.figure(figsize=(11, 6))
    for method_name, frame in method_frames.items():
        if value_column not in frame.columns:
            continue
        metric_frame = frame[["global_batch_index", value_column, phase_boundary_column]].dropna()
        if metric_frame.empty:
            continue
        plt.plot(
            metric_frame["global_batch_index"].to_numpy(),
            metric_frame[value_column].to_numpy(),
            label=method_name,
            linewidth=1.8,
        )

        # Draw one phase boundary marker per method; boundaries align when phases are comparable.
        boundaries = metric_frame[phase_boundary_column].dropna().unique()
        for boundary in boundaries:
            plt.axvline(x=float(boundary), color="gray", linestyle="--", alpha=0.15)

    plt.title(title)
    plt.xlabel("Global Batch Index")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_eval_method_comparison(
    eval_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    if metric not in eval_df.columns:
        return

    metric_df = eval_df[["method_name", metric]].dropna()
    if metric_df.empty:
        return

    metric_df = metric_df.sort_values(metric, ascending=False).reset_index(drop=True)
    x = np.arange(len(metric_df), dtype=np.int64)

    plt.figure(figsize=(10, 5))
    plt.bar(x, metric_df[metric].to_numpy(), width=0.65)
    plt.xticks(x, metric_df["method_name"].tolist(), rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(f"Post-phase2 evaluation comparison: {metric}")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def _init_or_get_wandb_for_plots(
    cfg: Dict[str, Any],
    run_label: str,
    metric_specs: List[Dict[str, str]],
):
    """Return (wandb_module_or_none, started_new_run)."""
    if not bool(cfg.get("report_to_wandb", False)):
        return None, False

    try:
        import wandb
    except ImportError:
        log.warning("report_to_wandb is enabled, but wandb is not installed; skipping plot logging")
        return None, False

    api_keys = _load_api_keys_from_config()
    wandb_api_key = api_keys.get("wandb_api_key")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)

    if wandb.run is not None:
        return wandb, False

    plot_run_name = cfg.get("phase_shift", {}).get("wandb_plots_name")
    if not plot_run_name:
        base_name = cfg.get("wandb_name")
        if base_name:
            plot_run_name = f"{base_name}_two_phase_plots"
        else:
            plot_run_name = f"{run_label}_plots"

    wandb.init(
        project=cfg.get("wandb_project", "multi-llm-wagering"),
        entity=cfg.get("wandb_entity", None),
        name=plot_run_name,
        config={
            "pipeline": "two_phase",
            "run_label": run_label,
            "metrics": [spec["output_metric"] for spec in metric_specs],
        },
        tags=["phase_shift", "comparison", "plots"],
    )
    return wandb, True


def _log_plots_to_wandb(
    wandb_logger,
    plots_dir: Path,
    metric_specs: List[Dict[str, str]],
    eval_df: pd.DataFrame,
    summary_path: Path,
) -> None:
    if wandb_logger is None:
        return

    payload: Dict[str, Any] = {}
    for spec in metric_specs:
        metric = spec["output_metric"]
        rolling_path = plots_dir / f"rolling_{metric}.png"
        phase_cum_path = plots_dir / f"phase_cumulative_{metric}.png"
        eval_path = plots_dir / f"eval_overall_{metric}.png"

        if rolling_path.exists():
            payload[f"phase_shift/plots/rolling/{metric}"] = wandb_logger.Image(str(rolling_path))
        if phase_cum_path.exists():
            payload[f"phase_shift/plots/phase_cumulative/{metric}"] = wandb_logger.Image(str(phase_cum_path))
        if eval_path.exists():
            payload[f"phase_shift/plots/eval_overall/{metric}"] = wandb_logger.Image(str(eval_path))

    if not eval_df.empty:
        payload["phase_shift/eval_method_comparison_table"] = wandb_logger.Table(dataframe=eval_df)

    if summary_path.exists() and hasattr(wandb_logger, "save"):
        wandb_logger.save(str(summary_path))

    if payload:
        wandb_logger.log(payload)


def _run_single_method(
    base_cfg: Dict[str, Any],
    run_label: str,
    method_cfg: Dict[str, Any],
    phase1_cfg: Dict[str, Any],
    phase2_cfg: Dict[str, Any],
    output_root: Path,
    run_evaluation: bool,
    device_override: Optional[str] = None,
    enable_method_wandb: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    method_name = str(method_cfg.get("name", "unknown_method"))
    method_slug = _slugify(method_name)
    method_output_dir = output_root / method_slug
    method_output_dir.mkdir(parents=True, exist_ok=True)

    # Only touch process-global wandb state when per-method wandb is enabled.
    # In parallel mode, per-method wandb is disabled and concurrent finish() calls
    # can serialize or stall workers before training even starts.
    if enable_method_wandb:
        _finish_active_wandb_run()

    with tempfile.TemporaryDirectory(prefix=f"phase_shift_{method_slug}_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        phase1_payload = _build_phase_config(
            base_cfg=base_cfg,
            phase_cfg=phase1_cfg,
            method_cfg=method_cfg,
            run_label=run_label,
            phase_name="phase1",
            resume_checkpoint=None,
            device_override=device_override,
            emit_resume_checkpoint=True,
            enable_method_wandb=enable_method_wandb,
        )
        phase1_cfg_path = tmp_dir_path / f"{method_slug}_phase1.yaml"
        _save_yaml(phase1_cfg_path, phase1_payload)

        log.info("[%s] Starting phase 1 training", method_name)
        phase1_results = train_main(config_path=str(phase1_cfg_path), calibration_path=None)
        phase1_checkpoint_dir = Path(phase1_results["checkpoint_path"]) 
        phase1_resume_ckpt = phase1_results.get("phase_transition_checkpoint_path")
        if phase1_resume_ckpt is None:
            phase1_resume_ckpt = str(_latest_transition_checkpoint(phase1_checkpoint_dir))

        phase2_payload = _build_phase_config(
            base_cfg=base_cfg,
            phase_cfg=phase2_cfg,
            method_cfg=method_cfg,
            run_label=run_label,
            phase_name="phase2",
            resume_checkpoint=str(phase1_resume_ckpt),
            device_override=device_override,
            emit_resume_checkpoint=False,
            enable_method_wandb=enable_method_wandb,
        )
        phase2_cfg_path = tmp_dir_path / f"{method_slug}_phase2.yaml"
        _save_yaml(phase2_cfg_path, phase2_payload)

        log.info("[%s] Starting phase 2 training (resume=%s)", method_name, phase1_resume_ckpt)
        phase2_results = train_main(config_path=str(phase2_cfg_path), calibration_path=None)
        phase2_checkpoint_dir = Path(phase2_results["checkpoint_path"]) 

        eval_results: Optional[Dict[str, Any]] = None
        if run_evaluation:
            log.info("[%s] Starting post-phase2 evaluation", method_name)
            eval_results = eval_main(
                config_path=str(phase2_cfg_path),
                checkpoint_path=str(phase2_checkpoint_dir),
                calibration_path=phase2_results.get("calibration_path"),
            )

    # Do not leak method-specific wandb run into subsequent methods.
    if enable_method_wandb:
        _finish_active_wandb_run()

    phase1_df = _load_batch_metrics(phase1_checkpoint_dir).copy()
    phase2_df = _load_batch_metrics(phase2_checkpoint_dir).copy()

    phase1_df["phase"] = "phase1"
    phase2_df["phase"] = "phase2"

    phase1_df = phase1_df.sort_values("global_step").reset_index(drop=True)
    phase2_df = phase2_df.sort_values("global_step").reset_index(drop=True)

    phase1_df["global_batch_index"] = np.arange(1, len(phase1_df) + 1, dtype=np.int64)
    phase2_df["global_batch_index"] = np.arange(len(phase1_df) + 1, len(phase1_df) + len(phase2_df) + 1, dtype=np.int64)

    boundary = len(phase1_df) + 0.5
    phase1_df["phase_boundary"] = boundary
    phase2_df["phase_boundary"] = boundary

    combined = pd.concat([phase1_df, phase2_df], ignore_index=True)
    combined["method_name"] = method_name

    return combined, {
        "method_name": method_name,
        "phase1_checkpoint_dir": str(phase1_checkpoint_dir),
        "phase2_checkpoint_dir": str(phase2_checkpoint_dir),
        "phase1_transition_checkpoint": str(phase1_resume_ckpt),
        "evaluation": eval_results,
        "evaluation_aggregated": _aggregate_eval_metrics(eval_results, DEFAULT_METRICS),
    }

def run_two_phase_pipeline(
    config_path: Path,
    device_override: Optional[str] = None,
    max_workers: int = 1,
) -> Dict[str, Any]:
    cfg = load_and_merge_configs(config_path)
    phase_shift_cfg = cfg.get("phase_shift", {})

    effective_device_override = device_override
    if effective_device_override is None:
        cfg_device = phase_shift_cfg.get("device") if isinstance(phase_shift_cfg, dict) else None
        if isinstance(cfg_device, str) and cfg_device.strip():
            effective_device_override = cfg_device.strip()

    if effective_device_override is not None:
        phase_shift_cfg["device"] = str(effective_device_override)
        cfg["phase_shift"] = phase_shift_cfg
        log.info("Using global phase-shift wagering device override: %s", effective_device_override)

    if "phase1" not in phase_shift_cfg or "phase2" not in phase_shift_cfg:
        raise ValueError("Config must define phase_shift.phase1 and phase_shift.phase2")

    phase1_cfg = phase_shift_cfg["phase1"]
    phase2_cfg = phase_shift_cfg["phase2"]
    methods_cfg = phase_shift_cfg.get("methods")

    if not methods_cfg:
        fallback_method = cfg.get("wagering_method")
        if fallback_method:
            methods_cfg = [copy.deepcopy(fallback_method)]
        else:
            raise ValueError(
                "Config must define either phase_shift.methods (preferred) "
                "or top-level wagering_method"
            )

    normalized_methods: List[Dict[str, Any]] = []
    for method_entry in methods_cfg:
        if isinstance(method_entry, str):
            normalized_methods.append({"name": method_entry, "config": {}})
        elif isinstance(method_entry, dict) and "name" in method_entry:
            normalized_methods.append(copy.deepcopy(method_entry))
        else:
            raise ValueError(f"Invalid method entry in phase_shift.methods: {method_entry}")

    rolling_window = int(phase_shift_cfg.get("rolling_window_batches", 20))
    raw_metrics = list(phase_shift_cfg.get("metrics", DEFAULT_METRICS))
    metric_aliases = {
        "mrr": "best_model_mrr",
    }
    metric_specs: List[Dict[str, str]] = []
    seen_metrics: set[Tuple[str, str]] = set()
    for metric_name in raw_metrics:
        output_metric = str(metric_name)
        source_metric = metric_aliases.get(output_metric, output_metric)
        metric_key = (source_metric, output_metric)
        if metric_key in seen_metrics:
            continue
        seen_metrics.add(metric_key)
        metric_specs.append({
            "source_metric": source_metric,
            "output_metric": output_metric,
        })

    run_evaluation = bool(phase_shift_cfg.get("evaluate_after_phase2", True))

    # Training/eval wandb logs from per-method runs share process-global state and
    # cannot be isolated safely across threads. In parallel mode, disable method
    # wandb logs and keep only pipeline-level plot logging.
    enable_method_wandb = True
    if max_workers > 1 and bool(cfg.get("report_to_wandb", False)):
        enable_method_wandb = False
        log.warning(
            "Disabling per-method wandb logging because max_workers=%d > 1. "
            "This avoids non-monotonic step collisions across concurrent method runs.",
            max_workers,
        )

    run_label = f"{_slugify(config_path.stem)}_two_phase"
    output_root = Path(phase_shift_cfg.get("output_dir", PROJECT_ROOT / "workdir" / "phase_shift_results" / run_label))
    output_root.mkdir(parents=True, exist_ok=True)
    method_cache_path = output_root / "method_metrics_cache.yaml"
    method_cache = _load_yaml(method_cache_path)
    cache_entries = method_cache.get("entries", {}) if isinstance(method_cache.get("entries"), dict) else {}

    wandb_logger, started_plot_run = _init_or_get_wandb_for_plots(
        cfg=cfg,
        run_label=run_label,
        metric_specs=metric_specs,
    )

    method_frames: Dict[str, pd.DataFrame] = {}
    method_summaries: List[Dict[str, Any]] = []
    failed_methods: List[Dict[str, str]] = []

    def _persist_method_cache() -> None:
        cache_payload = {
            "config_path": str(config_path),
            "run_label": run_label,
            "output_root": str(output_root),
            "entries": cache_entries,
        }
        _save_yaml(method_cache_path, cache_payload)

    def _finalize_method_result(
        method_name: str,
        method_slug: str,
        cache_key: str,
        frame: pd.DataFrame,
        summary: Dict[str, Any],
        method_frame_out: Path,
    ) -> None:
        for spec in metric_specs:
            source_metric = spec["source_metric"]
            output_metric = spec["output_metric"]
            if source_metric not in frame.columns:
                frame[f"rolling_{output_metric}"] = np.nan
                frame[f"phase_cum_{output_metric}"] = np.nan
                continue
            frame[f"rolling_{output_metric}"] = _rolling_mean(frame[source_metric], rolling_window)
            phase_cum = (
                frame.groupby("phase", group_keys=False)[source_metric]
                .apply(_phase_cumulative_mean)
            )
            frame[f"phase_cum_{output_metric}"] = phase_cum

        frame.to_csv(method_frame_out, index=False)

        cache_entries[method_slug] = {
            "cache_key": cache_key,
            "batch_metrics_path": str(method_frame_out),
            "summary": {
                "method_name": method_name,
                "phase1_checkpoint_dir": summary.get("phase1_checkpoint_dir"),
                "phase2_checkpoint_dir": summary.get("phase2_checkpoint_dir"),
                "phase1_transition_checkpoint": summary.get("phase1_transition_checkpoint"),
                "evaluation_aggregated": summary.get("evaluation_aggregated"),
                "evaluation": summary.get("evaluation"),
            },
        }
        _persist_method_cache()

        method_frames[method_name] = frame
        summary["batch_metrics_path"] = str(method_frame_out)
        method_summaries.append(summary)

    pending_methods: List[Dict[str, Any]] = []

    for method_cfg in normalized_methods:
        method_name = str(method_cfg["name"])
        method_slug = _slugify(method_name)
        method_frame_out = output_root / method_slug / "two_phase_batch_metrics.csv"
        cache_key = _build_method_cache_fingerprint(
            base_cfg=cfg,
            phase1_cfg=phase1_cfg,
            phase2_cfg=phase2_cfg,
            method_cfg=method_cfg,
            run_evaluation=run_evaluation,
        )

        cached_entry = cache_entries.get(method_slug, {})

        # Backfill cache index from prior partial runs where per-method outputs
        # exist but method_metrics_cache.yaml was not flushed yet.
        if (not isinstance(cached_entry, dict) or not cached_entry) and method_frame_out.is_file():
            cached_entry = {
                "cache_key": cache_key,
                "batch_metrics_path": str(method_frame_out),
                "summary": {
                    "method_name": method_name,
                    "phase1_checkpoint_dir": None,
                    "phase2_checkpoint_dir": None,
                    "phase1_transition_checkpoint": None,
                    "evaluation_aggregated": _aggregate_eval_metrics(None, DEFAULT_METRICS),
                    "evaluation": None,
                },
            }
            cache_entries[method_slug] = cached_entry
            _persist_method_cache()
            log.info("[%s] Bootstrapped method cache index from existing output %s", method_name, method_frame_out)

        cached_frame_path_str = cached_entry.get("batch_metrics_path") if isinstance(cached_entry, dict) else None
        cached_frame_path = Path(cached_frame_path_str) if isinstance(cached_frame_path_str, str) and cached_frame_path_str else None

        if (
            isinstance(cached_entry, dict)
            and cached_entry.get("cache_key") == cache_key
            and cached_frame_path is not None
            and cached_frame_path.is_file()
        ):
            log.info("[%s] Cache hit: loading existing metrics from %s", method_name, cached_frame_path)
            frame = pd.read_csv(cached_frame_path)
            summary = copy.deepcopy(cached_entry.get("summary", {}))
            summary["method_name"] = method_name
            summary["batch_metrics_path"] = str(cached_frame_path)
            summary["cached"] = True
            _finalize_method_result(
                method_name=method_name,
                method_slug=method_slug,
                cache_key=cache_key,
                frame=frame,
                summary=summary,
                method_frame_out=method_frame_out,
            )
        else:
            pending_methods.append(
                {
                    "method_cfg": method_cfg,
                    "method_name": method_name,
                    "method_slug": method_slug,
                    "cache_key": cache_key,
                    "method_frame_out": method_frame_out,
                }
            )

    if pending_methods:
        workers = max(1, int(max_workers))
        workers = min(workers, len(pending_methods))

        if workers > 1:
            log.info(
                "Executing %d uncached method(s) in parallel with max_workers=%d",
                len(pending_methods),
                workers,
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_method,
                        base_cfg=cfg,
                        run_label=run_label,
                        method_cfg=job["method_cfg"],
                        phase1_cfg=phase1_cfg,
                        phase2_cfg=phase2_cfg,
                        output_root=output_root,
                        run_evaluation=run_evaluation,
                        device_override=effective_device_override,
                        enable_method_wandb=enable_method_wandb,
                    ): job
                    for job in pending_methods
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        frame, summary = future.result()
                    except Exception as e:
                        method_name = str(job.get("method_name", "unknown_method"))
                        error_msg = f"{type(e).__name__}: {e}"
                        log.exception(
                            "[%s] Method execution failed in parallel worker; continuing with remaining methods",
                            method_name,
                        )
                        method_summaries.append(
                            {
                                "method_name": method_name,
                                "phase1_checkpoint_dir": None,
                                "phase2_checkpoint_dir": None,
                                "phase1_transition_checkpoint": None,
                                "evaluation": None,
                                "evaluation_aggregated": _aggregate_eval_metrics(None, DEFAULT_METRICS),
                                "batch_metrics_path": None,
                                "cached": False,
                                "failed": True,
                                "error": error_msg,
                            }
                        )
                        failed_methods.append(
                            {
                                "method_name": method_name,
                                "error": error_msg,
                            }
                        )
                        continue
                    summary["cached"] = False
                    _finalize_method_result(
                        method_name=job["method_name"],
                        method_slug=job["method_slug"],
                        cache_key=job["cache_key"],
                        frame=frame,
                        summary=summary,
                        method_frame_out=job["method_frame_out"],
                    )
        else:
            for job in pending_methods:
                frame, summary = _run_single_method(
                    base_cfg=cfg,
                    run_label=run_label,
                    method_cfg=job["method_cfg"],
                    phase1_cfg=phase1_cfg,
                    phase2_cfg=phase2_cfg,
                    output_root=output_root,
                    run_evaluation=run_evaluation,
                    device_override=effective_device_override,
                    enable_method_wandb=enable_method_wandb,
                )
                summary["cached"] = False
                _finalize_method_result(
                    method_name=job["method_name"],
                    method_slug=job["method_slug"],
                    cache_key=job["cache_key"],
                    frame=frame,
                    summary=summary,
                    method_frame_out=job["method_frame_out"],
                )

    _persist_method_cache()

    # Combined plots across methods.
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for spec in metric_specs:
        output_metric = spec["output_metric"]
        _plot_metric_over_time(
            method_frames=method_frames,
            metric=output_metric,
            out_path=plots_dir / f"rolling_{output_metric}.png",
            y_label=f"Rolling {output_metric} (window={rolling_window})",
            title=f"Rolling {output_metric} across methods",
            value_column=f"rolling_{output_metric}",
            phase_boundary_column="phase_boundary",
        )

    eval_rows: List[Dict[str, Any]] = []
    for method_summary in method_summaries:
        row = {"method_name": method_summary.get("method_name")}
        eval_agg = method_summary.get("evaluation_aggregated", {})
        if isinstance(eval_agg, dict):
            for spec in metric_specs:
                source_metric = spec["source_metric"]
                output_metric = spec["output_metric"]
                row[output_metric] = eval_agg.get(source_metric)
        eval_rows.append(row)

    eval_df = pd.DataFrame(eval_rows)
    eval_out = output_root / "eval_method_comparison.csv"
    eval_df.to_csv(eval_out, index=False)

    for spec in metric_specs:
        output_metric = spec["output_metric"]
        _plot_eval_method_comparison(
            eval_df=eval_df,
            metric=output_metric,
            out_path=plots_dir / f"eval_overall_{output_metric}.png",
        )
        _plot_metric_over_time(
            method_frames=method_frames,
            metric=output_metric,
            out_path=plots_dir / f"phase_cumulative_{output_metric}.png",
            y_label=f"Phase cumulative {output_metric}",
            title=f"Phase-reset cumulative {output_metric} across methods",
            value_column=f"phase_cum_{output_metric}",
            phase_boundary_column="phase_boundary",
        )

    summary_payload = {
        "config_path": str(config_path),
        "output_root": str(output_root),
        "rolling_window_batches": rolling_window,
        "metrics": [spec["output_metric"] for spec in metric_specs],
        "eval_method_comparison_csv": str(eval_out),
        "methods": method_summaries,
        "failed_methods": failed_methods,
    }
    summary_path = output_root / "summary.yaml"
    _save_yaml(summary_path, summary_payload)

    _log_plots_to_wandb(
        wandb_logger=wandb_logger,
        plots_dir=plots_dir,
        metric_specs=metric_specs,
        eval_df=eval_df,
        summary_path=summary_path,
    )

    if wandb_logger is not None and started_plot_run and hasattr(wandb_logger, "finish"):
        wandb_logger.finish()

    if failed_methods:
        failed_names = ", ".join(m["method_name"] for m in failed_methods)
        raise RuntimeError(
            f"Two-phase run completed with {len(failed_methods)} failed method(s): {failed_names}. "
            f"See summary at {summary_path} for details."
        )

    log.info("Two-phase run complete. Outputs: %s", output_root)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-phase wagering pipeline")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Number of wagering methods to execute concurrently. "
            "Use 1 for sequential execution (default)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device override for all phase-shift methods (for example: "
            "cuda:0, cuda:1, cpu). Overrides phase_shift.device in config."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if int(args.max_workers) <= 0:
        raise ValueError("--max-workers must be > 0")

    run_two_phase_pipeline(
        config_path,
        device_override=args.device,
        max_workers=int(args.max_workers),
    )


if __name__ == "__main__":
    main()
