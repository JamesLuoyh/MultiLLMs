#!/usr/bin/env python3
"""
Hyperparameter search for wagering training configs (learning rate only).

Runs training for each (learning_rate, seed) pair with wandb disabled,
aggregates validation-set metrics across repeats, and prints the best LR.

Example:
  python scripts/wagering_lr_search.py \
    --config examples/configs/wagering_training/mse_br_wagers_v2_pubmedqa.yaml \
    --n-repeats 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
DISK_ROOT = PROJECT_ROOT


def _ensure_project_importable() -> None:
    """Match scripts/wagering_train.py so `import wagering` works in worker processes."""
    root = str(PROJECT_ROOT)
    src = str(SRC_PATH)
    if src not in sys.path:
        sys.path.insert(0, src)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_and_merge_config(config_path: Path) -> Dict[str, Any]:
    _ensure_project_importable()
    from wagering.utils.config_utils import load_and_merge_configs

    return load_and_merge_configs(config_path)


def _load_script_main(script_filename: str):
    """Load `main` from a repo script file (avoids runpy breaking package imports)."""
    _ensure_project_importable()
    path = PROJECT_ROOT / "scripts" / script_filename
    mod_name = f"_wagering_lr_search_{script_filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "main")


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _parse_gpus(gpus: str) -> List[int]:
    out: List[int] = []
    for part in gpus.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No GPUs provided. Example: --gpus 0,1,2,3")
    return out


def _coarse_lrs_fixed() -> List[float]:
    # Exactly: 1e-6, 1e-5, ..., 1e-1
    return [10.0 ** (-k) for k in range(7, 3, -1)]


def _log10_int(x: float) -> int:
    return int(round(-math.log10(x)))


def _fine_lrs_around(best_lr: float) -> List[float]:
    # If best is 1e-i, explore around it on the same log decade:
    #   5e-(i+1) < 1e-i < 5e-i
    # Example: best=1e-4 => [5e-5, 5e-4]
    i = _log10_int(best_lr)
    return [5.0 * (10.0 ** (-(i + 1))), 5.0 * (10.0 ** (-i))]


def _ensure_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x)
    raise TypeError(f"Cannot convert to float: {type(x)}")


def _safe_yaml_dump(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_objective_from_train_results(
    train_results: Dict[str, Any],
    metric: str,
) -> Optional[float]:
    if not isinstance(train_results, dict):
        return None

    # Preferred path: explicitly exported final validation metrics.
    val_metrics = train_results.get("final_validation_metrics")
    if isinstance(val_metrics, dict) and metric in val_metrics:
        try:
            return _ensure_float(val_metrics.get(metric))
        except Exception:
            pass

    # Backward-compatible fallback for common metric names.
    fallback_key = f"final_validation_{metric}"
    if fallback_key in train_results:
        try:
            return _ensure_float(train_results.get(fallback_key))
        except Exception:
            pass

    return None


def _objective_is_higher_better(metric: str) -> bool:
    # Common convention: maximize accuracy/auc, minimize loss-like metrics.
    if metric.lower() in {"accuracy", "auc", "meta_acc", "meta_auc", "kendall_tau", "best_model_mrr"}:
        return True
    return False


@dataclass(frozen=True)
class JobSpec:
    lr: float
    seed: int
    stage: str  # "coarse" or "fine"


@dataclass
class JobResult:
    lr: float
    seed: int
    stage: str
    gpu: int
    checkpoint_path: Optional[str]
    objective: Optional[float]
    metric: str
    error: Optional[str]


def _make_derived_config(
    base_config_path: Path,
    lr: float,
    seed: int,
    workdir: Path,
    run_id: str,
) -> Tuple[Path, Dict[str, Any]]:
    args = _load_and_merge_config(base_config_path)

    # Disable wandb during search (as requested).
    args["report_to_wandb"] = False

    # Force stable dataset splits and override randomization only via seed.
    args["shuffle_seed"] = int(seed)
    args["seed"] = int(seed)

    # Ensure we don't auto-resume across different LRs/seeds.
    args["auto_resume"] = False
    args.pop("resume_from_checkpoint", None)

    # LR search only needs final validation metrics; avoid writing bulky checkpoints.
    args["save_epoch_checkpoints"] = False
    args["save_final_checkpoint"] = False

    # Unique per (lr, seed) so parallel jobs never share a checkpoint directory.
    job_key = f"lr_{lr:.10g}_seed_{seed}".replace("+", "")
    args["checkpoint_base_dir"] = str(workdir / "checkpoints" / run_id / job_key)

    # Only override learning rate (hyperparameter under search).
    wm = args.get("wagering_method", {})
    wm_cfg = wm.get("config", {})
    wm_cfg["learning_rate"] = float(lr)
    wm["config"] = wm_cfg
    args["wagering_method"] = wm

    # Put config on disk (train/eval scripts accept a path).
    cfg_out = workdir / "derived_configs" / run_id / f"lr_{lr:.0e}" / f"seed_{seed}.yaml"
    _safe_yaml_dump(args, cfg_out)
    return cfg_out, args


def _run_train_eval_on_gpu(
    *,
    gpu: int,
    base_config_path: str,
    lr: float,
    seed: int,
    metric: str,
    stage: str,
    workdir: str,
    run_id: str,
    cleanup_artifacts: bool,
) -> JobResult:
    try:
        # Isolate GPU and disable wandb at env level too.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_SILENT"] = "true"

        # Reduce CPU oversubscription when running many processes.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        base_config_path_p = Path(base_config_path)
        workdir_p = Path(workdir)
        _ensure_project_importable()
        derived_config_path, derived_args = _make_derived_config(
            base_config_path=base_config_path_p,
            lr=lr,
            seed=seed,
            workdir=workdir_p,
            run_id=run_id,
        )

        train_main = _load_script_main("wagering_train.py")

        train_res = train_main(str(derived_config_path))
        checkpoint_path = train_res.get("checkpoint_path") if isinstance(train_res, dict) else None
        calibration_path = train_res.get("calibration_path") if isinstance(train_res, dict) else None

        # Ensure eval uses the fresh checkpoint.
        if checkpoint_path is None:
            raise RuntimeError("Training did not return checkpoint_path.")

        # Also write checkpoint_path into config for any downstream code.
        try:
            derived_args["checkpoint_path"] = str(checkpoint_path)
            _safe_yaml_dump(derived_args, derived_config_path)
        except Exception:
            pass

        objective = _extract_objective_from_train_results(train_res, metric=metric)

        result = JobResult(
            lr=lr,
            seed=seed,
            stage=stage,
            gpu=gpu,
            checkpoint_path=str(checkpoint_path),
            objective=objective,
            metric=metric,
            error=None,
        )

        if cleanup_artifacts:
            # Remove per-job artifacts to keep disk usage bounded during large sweeps.
            try:
                if checkpoint_path is not None:
                    shutil.rmtree(str(checkpoint_path), ignore_errors=True)
            except Exception:
                pass
            try:
                derived_config_path.unlink(missing_ok=True)
                lr_dir = derived_config_path.parent
                if lr_dir.exists() and not any(lr_dir.iterdir()):
                    lr_dir.rmdir()
            except Exception:
                pass

        return result
    except Exception as e:
        return JobResult(
            lr=lr,
            seed=seed,
            stage=stage,
            gpu=gpu,
            checkpoint_path=None,
            objective=None,
            metric=metric,
            error=str(e),
        )


def _schedule_jobs(
    jobs: List[JobSpec],
    *,
    base_config_path: Path,
    metric: str,
    gpus: List[int],
    procs_per_gpu: int,
    workdir: Path,
    run_id: str,
    cleanup_artifacts: bool,
) -> List[JobResult]:
    gpu_slots: "queue.Queue[int]" = queue.Queue()
    for g in gpus:
        for _ in range(procs_per_gpu):
            gpu_slots.put(g)

    results: List[JobResult] = []
    max_workers = len(gpus) * procs_per_gpu

    def submit_one(executor: ProcessPoolExecutor, job: JobSpec):
        gpu = gpu_slots.get()
        fut = executor.submit(
            _run_train_eval_on_gpu,
            gpu=gpu,
            base_config_path=str(base_config_path),
            lr=job.lr,
            seed=job.seed,
            metric=metric,
            stage=job.stage,
            workdir=str(workdir),
            run_id=run_id,
            cleanup_artifacts=cleanup_artifacts,
        )
        fut._wagering_gpu = gpu  # type: ignore[attr-defined]
        fut._wagering_job = job  # type: ignore[attr-defined]
        return fut

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [submit_one(ex, j) for j in jobs]
        for fut in as_completed(futures):
            gpu = getattr(fut, "_wagering_gpu", None)
            if gpu is not None:
                gpu_slots.put(int(gpu))
            res = fut.result()
            results.append(res)
            # Stream progress as JSON lines for easy parsing.
            print(json.dumps(res.__dict__, sort_keys=True), flush=True)
    return results


def _pick_best_lr(
    results: List[JobResult],
    *,
    metric: str,
) -> Tuple[Optional[float], Dict[float, List[JobResult]]]:
    by_lr: Dict[float, List[JobResult]] = {}
    for r in results:
        by_lr.setdefault(r.lr, []).append(r)

    higher_better = _objective_is_higher_better(metric)
    best_lr: Optional[float] = None
    best_score: Optional[float] = None

    for lr, runs in by_lr.items():
        vals = [rr.objective for rr in runs if rr.error is None and rr.objective is not None]
        score = _mean([v for v in vals if v is not None]) if vals else None
        if score is None:
            continue
        if best_score is None:
            best_lr, best_score = lr, score
            continue
        if higher_better and score > best_score:
            best_lr, best_score = lr, score
        if (not higher_better) and score < best_score:
            best_lr, best_score = lr, score

    return best_lr, by_lr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str, help="Path to base YAML config.")
    p.add_argument("--n-repeats", required=True, type=int, help="Number of random seeds per LR.")
    p.add_argument("--seed-start", default=2025, type=int, help="First seed (default: 2025).")
    p.add_argument("--metric", default="accuracy", type=str, help="Eval metric to optimize (default: accuracy).")
    p.add_argument("--gpus", default="0,1,2,3", type=str, help="Comma-separated GPU ids (default: 0,1,2,3).")
    p.add_argument("--procs-per-gpu", default=1, type=int, help="Parallel processes per GPU (default: 5).")
    p.add_argument(
        "--workdir",
        default=str(DISK_ROOT / "workdir" / "lr_search"),
        type=str,
        help="Directory for derived configs + checkpoints.",
    )
    p.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep per-job checkpoints/configs (disabled by default to save disk).",
    )
    args = p.parse_args()

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Config not found: {base_config_path}")

    gpus = _parse_gpus(args.gpus)
    workdir = Path(args.workdir)
    run_id = _now_run_id()
    cleanup_artifacts = not bool(args.keep_run_artifacts)

    seeds = [int(args.seed_start + i) for i in range(int(args.n_repeats))]
    coarse_lrs = _coarse_lrs_fixed()

    coarse_jobs = [JobSpec(lr=lr, seed=s, stage="coarse") for lr in coarse_lrs for s in seeds]
    coarse_results = _schedule_jobs(
        coarse_jobs,
        base_config_path=base_config_path,
        metric=args.metric,
        gpus=gpus,
        procs_per_gpu=int(args.procs_per_gpu),
        workdir=workdir,
        run_id=run_id,
        cleanup_artifacts=cleanup_artifacts,
    )

    best_lr, _ = _pick_best_lr(coarse_results, metric=args.metric)
    if best_lr is None:
        raise RuntimeError("Could not determine best LR from coarse sweep (no successful runs).")

    fine_lrs = [lr for lr in _fine_lrs_around(best_lr) if lr not in set(coarse_lrs)]
    fine_results: List[JobResult] = []
    if fine_lrs:
        fine_jobs = [JobSpec(lr=lr, seed=s, stage="fine") for lr in fine_lrs for s in seeds]
        fine_results = _schedule_jobs(
            fine_jobs,
            base_config_path=base_config_path,
            metric=args.metric,
            gpus=gpus,
            procs_per_gpu=int(args.procs_per_gpu),
            workdir=workdir,
            run_id=run_id,
            cleanup_artifacts=cleanup_artifacts,
        )

    all_results = coarse_results + fine_results
    final_best_lr, by_lr = _pick_best_lr(all_results, metric=args.metric)
    if final_best_lr is None:
        raise RuntimeError("Could not determine best LR from full sweep (no successful runs).")

    # Print a concise summary and final best LR.
    print("\n=== LR search summary ===", flush=True)
    higher_better = _objective_is_higher_better(args.metric)
    direction = "maximize" if higher_better else "minimize"
    print(f"Objective: {args.metric} ({direction})", flush=True)

    for lr in sorted(by_lr.keys()):
        runs = by_lr[lr]
        ok = [r.objective for r in runs if r.error is None and r.objective is not None]
        mean_obj = _mean([v for v in ok if v is not None])
        n_ok = len(ok)
        n_total = len(runs)
        mean_str = "N/A" if mean_obj is None else f"{mean_obj:.6f}"
        print(f"lr={lr:.0e}  mean_{args.metric}={mean_str}  ok={n_ok}/{n_total}", flush=True)

    print("\n=== BEST LEARNING RATE (validation set) ===", flush=True)
    print(f"{final_best_lr:.6g}", flush=True)


if __name__ == "__main__":
    main()

