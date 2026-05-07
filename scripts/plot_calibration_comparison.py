#!/usr/bin/env python3
"""
Run two calibration settings and generate requested plots.

Settings:
  (A) Uncalibrated: calibrated=false
  (B) Gemma2-only calibrated: calibrated=true + calibration.apply_to_model_indices=[0]

Outputs:
  - Training plots (per model): mean wager + mean net payout over batches (smoothed window=100)
  - Test bar plots (ID test only): avg wager + avg net payout per model (pairs of bars)

Default --out-dir is under /research/projects/ecoai/yl2310/MultiLLMs/artifacts/.
If --out-dir is relative (e.g. artifacts/foo), it is resolved under MULTILLMS_OUTPUT_ROOT, not repo cwd.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODEL_DISPLAY = {
    "gemma_2_9b_it": "Gemma2",
    "llama3_1_8b_instruct": "Llama3.1",
    "llama3_aloe_8b_alpha": "Aloe",
    "biomistral_7b": "Biomistral",
}


@dataclass(frozen=True)
class RunPaths:
    checkpoint_dir: Path
    batch_metrics_csv: Path
    pipeline_summary_json: Path
    eval_results_json: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Plots / artifacts default here (not under repo/home) to avoid quota issues.
# Override with env MULTILLMS_OUTPUT_ROOT if needed.
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("MULTILLMS_OUTPUT_ROOT", "/research/projects/ecoai/yl2310/MultiLLMs")
).expanduser()


def _resolve_output_dir(path_str: str) -> Path:
    """Absolute paths as-is; relative paths are under DEFAULT_OUTPUT_ROOT (not cwd)."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (DEFAULT_OUTPUT_ROOT / p).resolve()


# Ensure repo packages are importable when running as a script.
PROJECT_ROOT = _repo_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _venv_python() -> Path:
    root = _repo_root()
    py = root / ".venv" / "bin" / "python"
    if not py.exists():
        raise RuntimeError(f"Expected venv python at {py}. Create .venv per repo rules.")
    return py


def _load_yaml(path: Path) -> Dict:
    import yaml

    with path.open("r") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _model_names_from_config(cfg: Dict) -> List[str]:
    models = cfg.get("models") or []
    out = []
    for m in models:
        p = str(m.get("path", ""))
        out.append(Path(p).name.replace("-", "_"))
    return out


def _display_names_for_models(cfg: Dict) -> List[str]:
    raw = _model_names_from_config(cfg)
    return [MODEL_DISPLAY.get(n, n) for n in raw]


def _infer_primary_test_display_name(cfg: Dict) -> str:
    tds = cfg.get("test_datasets") or []
    if not tds or not isinstance(tds, list) or not isinstance(tds[0], dict):
        raise ValueError("Expected config to define test_datasets as a list of dicts.")
    dn = tds[0].get("display_name")
    if not isinstance(dn, str) or not dn.strip():
        raise ValueError("Expected test_datasets[0].display_name to be a non-empty string.")
    return dn.strip()


def _make_setting_configs(base_cfg_path: Path, *, num_epochs: int = 1) -> Tuple[Path, Path]:
    base_cfg = _load_yaml(base_cfg_path)

    cfg_a = dict(base_cfg)
    cfg_a["calibrated"] = False
    cfg_a["num_epochs"] = int(num_epochs)

    cfg_b = dict(base_cfg)
    cfg_b["calibrated"] = True
    cfg_b["num_epochs"] = int(num_epochs)
    cal = dict(cfg_b.get("calibration") or {})
    # Apply calibration only to Gemma2 (index 0 in the 4-model configs provided).
    cal["apply_to_model_indices"] = [0]
    cfg_b["calibration"] = cal

    # Write in the same directory as the original config so relative `_include_*` paths resolve.
    out_dir = base_cfg_path.parent
    a_path = out_dir / (base_cfg_path.stem + "__uncalibrated.yaml")
    b_path = out_dir / (base_cfg_path.stem + "__gemma2_calibrated.yaml")
    _dump_yaml(cfg_a, a_path)
    _dump_yaml(cfg_b, b_path)
    return a_path, b_path


_CKPT_RE = re.compile(r"Checkpoint directory:\s*(?P<path>/.+)")


def _run_pipeline(config_path: Path, *, extra_env: Optional[Dict[str, str]] = None) -> RunPaths:
    root = _repo_root()
    py = _venv_python()

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [str(py), str(root / "scripts" / "wagering_pipeline.py"), str(config_path)]
    proc = subprocess.run(cmd, cwd=str(root), env=env, text=True, capture_output=True)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline failed for {config_path}.\n\n{out[-8000:]}")

    m = _CKPT_RE.search(out)
    if not m:
        raise RuntimeError(
            "Could not find checkpoint directory in pipeline output. "
            "Expected a line like 'Checkpoint directory: /path/to/dir'."
        )

    ckpt_dir = Path(m.group("path")).expanduser().resolve()
    batch_csv = ckpt_dir / "batch_metrics.csv"
    summary_json = ckpt_dir / "pipeline_artifacts" / "pipeline.summary.json"
    eval_json = ckpt_dir / "pipeline_artifacts" / "eval.results.json"

    if not batch_csv.exists():
        raise FileNotFoundError(f"Missing {batch_csv} (needed for training plots).")
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing {summary_json} (pipeline artifacts).")
    if not eval_json.exists():
        raise FileNotFoundError(f"Missing {eval_json} (pipeline artifacts).")

    return RunPaths(
        checkpoint_dir=ckpt_dir,
        batch_metrics_csv=batch_csv,
        pipeline_summary_json=summary_json,
        eval_results_json=eval_json,
    )


def mean_std_ci95(values: List[float]) -> Tuple[float, Optional[float], Optional[float], int]:
    """Mean and 95%% CI for the mean (Student's t). Returns (mean, ci_low, ci_high, n)."""
    from scipy import stats

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return float("nan"), None, None, 0
    mean = float(np.mean(arr))
    if n < 2:
        return mean, None, None, n
    std = float(np.std(arr, ddof=1))
    sem = std / float(np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    margin = tcrit * sem
    return mean, mean - margin, mean + margin, n


def discover_run_paths_under_repeat_arm(arm_dir: Path) -> List[RunPaths]:
    """Collect RunPaths for repeat_*/**/pipeline_artifacts/eval.results.json under one arm directory."""
    out: List[RunPaths] = []
    for rep_dir in sorted(arm_dir.glob("repeat_*")):
        if not rep_dir.is_dir():
            continue
        for eval_json in sorted(rep_dir.glob("**/pipeline_artifacts/eval.results.json")):
            ckpt_dir = eval_json.parent.parent
            batch_csv = ckpt_dir / "batch_metrics.csv"
            summary_json = ckpt_dir / "pipeline_artifacts" / "pipeline.summary.json"
            if batch_csv.is_file() and summary_json.is_file():
                out.append(
                    RunPaths(
                        checkpoint_dir=ckpt_dir.resolve(),
                        batch_metrics_csv=batch_csv,
                        pipeline_summary_json=summary_json,
                        eval_results_json=eval_json.resolve(),
                    )
                )
                break
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    w = int(window)
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    # pad to preserve length
    pad_left = w - 1
    xp = np.pad(x, (pad_left, 0), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def _load_eval_dataset_result(eval_results_json: Path, dataset_name: str) -> Dict:
    with eval_results_json.open("r") as f:
        res = json.load(f)
    if dataset_name not in res:
        raise KeyError(f"Dataset {dataset_name!r} not found in eval.results.json keys={list(res.keys())[:20]}")
    ds = res[dataset_name]
    if not isinstance(ds, dict):
        raise ValueError(f"Unexpected eval result shape for {dataset_name}: {type(ds).__name__}")
    return ds


def _plot_training_curves(
    *,
    out_dir: Path,
    title_prefix: str,
    model_display_names: List[str],
    run_a: RunPaths,
    run_b: RunPaths,
    window: int,
) -> None:
    import matplotlib.pyplot as plt

    df_a = pd.read_csv(run_a.batch_metrics_csv)
    df_b = pd.read_csv(run_b.batch_metrics_csv)

    n_models = len(model_display_names)
    x_a = np.arange(len(df_a), dtype=np.int64)
    x_b = np.arange(len(df_b), dtype=np.int64)

    # Plot set 1: mean wager per batch
    for mi in range(n_models):
        col = f"wager_model_{mi}"
        if col not in df_a.columns or col not in df_b.columns:
            raise KeyError(f"Missing {col} in batch_metrics.csv. Re-run with updated trainer logging.")
        ya = _rolling_mean(df_a[col].to_numpy(), window)
        yb = _rolling_mean(df_b[col].to_numpy(), window)

        plt.figure(figsize=(10, 4))
        plt.plot(x_a, ya, label="Uncalibrated", linewidth=1.6)
        plt.plot(x_b, yb, label="Gemma2 calibrated", linewidth=1.6)
        plt.xlabel("Training batch")
        plt.ylabel("Mean wager")
        plt.title(f"{title_prefix} — {model_display_names[mi]}: mean wager (running avg window={window})")
        plt.legend()
        plt.tight_layout()
        (out_dir / "plot_set_1").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "plot_set_1" / f"{title_prefix}__mean_wager__{model_display_names[mi]}.png", dpi=200)
        plt.close()

    # Plot set 2: mean net payout per batch
    for mi in range(n_models):
        col = f"net_payout_model_{mi}"
        if col not in df_a.columns or col not in df_b.columns:
            raise KeyError(f"Missing {col} in batch_metrics.csv. Re-run with updated trainer logging.")
        ya = _rolling_mean(df_a[col].to_numpy(), window)
        yb = _rolling_mean(df_b[col].to_numpy(), window)

        plt.figure(figsize=(10, 4))
        plt.plot(x_a, ya, label="Uncalibrated", linewidth=1.6)
        plt.plot(x_b, yb, label="Gemma2 calibrated", linewidth=1.6)
        plt.xlabel("Training batch")
        plt.ylabel("Mean net payout")
        plt.title(f"{title_prefix} — {model_display_names[mi]}: mean net payout (running avg window={window})")
        plt.legend()
        plt.tight_layout()
        (out_dir / "plot_set_2").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "plot_set_2" / f"{title_prefix}__mean_net_payout__{model_display_names[mi]}.png", dpi=200)
        plt.close()


def _plot_test_bars(
    *,
    out_dir: Path,
    title_prefix: str,
    model_display_names: List[str],
    eval_dataset_name: str,
    run_a: RunPaths,
    run_b: RunPaths,
) -> None:
    import matplotlib.pyplot as plt

    def _annotate_bars(ax, bars) -> None:
        for b in bars:
            try:
                h = float(b.get_height())
            except Exception:
                continue
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    a = _load_eval_dataset_result(run_a.eval_results_json, eval_dataset_name)
    b = _load_eval_dataset_result(run_b.eval_results_json, eval_dataset_name)

    avg_wager_a = np.asarray(a.get("avg_wager_per_model"), dtype=np.float64)
    avg_wager_b = np.asarray(b.get("avg_wager_per_model"), dtype=np.float64)
    avg_payout_a = np.asarray(a.get("avg_net_payout_per_model"), dtype=np.float64)
    avg_payout_b = np.asarray(b.get("avg_net_payout_per_model"), dtype=np.float64)
    avg_sigmoid_a = np.asarray(a.get("avg_sigmoid_wager_per_model"), dtype=np.float64)
    avg_sigmoid_b = np.asarray(b.get("avg_sigmoid_wager_per_model"), dtype=np.float64)

    if avg_wager_a.size == 0 or avg_wager_b.size == 0:
        raise ValueError("Missing avg_wager_per_model in eval results. Re-run with updated evaluator logging.")
    if avg_payout_a.size == 0 or avg_payout_b.size == 0:
        raise ValueError("Missing avg_net_payout_per_model in eval results. Ensure wagering method returns total_payout.")
    if avg_sigmoid_a.size == 0 or avg_sigmoid_b.size == 0:
        raise ValueError("Missing avg_sigmoid_wager_per_model in eval results. Re-run with updated evaluator logging.")

    n_models = len(model_display_names)
    if avg_wager_a.shape[0] != n_models or avg_wager_b.shape[0] != n_models:
        raise ValueError("Mismatch between models in config and eval results (avg_wager_per_model length).")

    x = np.arange(n_models)
    width = 0.35

    # Plot set 3: avg wager over ID test set
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    bars_a = ax.bar(x - width / 2, avg_wager_a, width, label="All Uncalibrated")
    bars_b = ax.bar(x + width / 2, avg_wager_b, width, label="Gemma2 Calibrated")
    ax.set_xticks(x, model_display_names)
    ax.set_ylabel("Average wager (normalized)")
    ax.set_title(f"{title_prefix} — Average Normalized Wagers")
    ax.legend()
    _annotate_bars(ax, bars_a)
    _annotate_bars(ax, bars_b)
    plt.tight_layout()
    (out_dir / "plot_set_3").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_set_3" / f"{title_prefix}__avg_wager_test.png", dpi=200)
    plt.close()

    # Plot set 4: avg net payout over ID test set
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    bars_a = ax.bar(x - width / 2, avg_payout_a, width, label="All Uncalibrated")
    bars_b = ax.bar(x + width / 2, avg_payout_b, width, label="Gemma2 Calibrated")
    ax.set_xticks(x, model_display_names)
    ax.set_ylabel("Average net payout")
    ax.set_title(f"{title_prefix} — Average Net Payout")
    ax.legend()
    _annotate_bars(ax, bars_a)
    _annotate_bars(ax, bars_b)
    plt.tight_layout()
    (out_dir / "plot_set_4").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_set_4" / f"{title_prefix}__avg_net_payout_test.png", dpi=200)
    plt.close()

    # Plot set 5: avg unnormalized wager (sigmoid_wagers) over ID test set
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    bars_a = ax.bar(x - width / 2, avg_sigmoid_a, width, label="All Uncalibrated")
    bars_b = ax.bar(x + width / 2, avg_sigmoid_b, width, label="Gemma2 Calibrated")
    ax.set_xticks(x, model_display_names)
    ax.set_ylabel("Average wager (unnormalized)")
    ax.set_title(f"{title_prefix} — Average Wagers")
    ax.legend()
    _annotate_bars(ax, bars_a)
    _annotate_bars(ax, bars_b)
    plt.tight_layout()
    (out_dir / "plot_set_5").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "plot_set_5" / f"{title_prefix}__avg_unnormalized_wager_test.png", dpi=200)
    plt.close()


def _series_mean_ci_per_step(
    series_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stack ragged series to min length; per-step mean and 95%% CI across runs."""
    if not series_list:
        return np.array([]), np.array([]), np.array([]), 0
    L = min(int(s.shape[0]) for s in series_list)
    if L <= 0:
        return np.array([]), np.array([]), np.array([]), 0
    mat = np.stack([s[:L].astype(np.float64) for s in series_list], axis=0)
    n = mat.shape[0]
    mean = np.mean(mat, axis=0)
    if n < 2:
        return mean, np.full(L, np.nan), np.full(L, np.nan), n
    from scipy import stats

    std = np.std(mat, axis=0, ddof=1)
    sem = std / float(np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    margin = tcrit * sem
    return mean, mean - margin, mean + margin, n


def _plot_training_curves_with_ci(
    *,
    out_dir: Path,
    title_prefix: str,
    model_display_names: List[str],
    runs_a: List[RunPaths],
    runs_b: List[RunPaths],
    window: int,
) -> None:
    import matplotlib.pyplot as plt

    n_models = len(model_display_names)

    def _collect_smoothed(runs: List[RunPaths], col: str) -> List[np.ndarray]:
        out_series: List[np.ndarray] = []
        for r in runs:
            df = pd.read_csv(r.batch_metrics_csv)
            if col not in df.columns:
                raise KeyError(f"Missing {col} in {r.batch_metrics_csv}")
            out_series.append(_rolling_mean(df[col].to_numpy(), window))
        return out_series

    # Plot set 1: mean wager per batch + 95% CI (no fill)
    for mi in range(n_models):
        col = f"wager_model_{mi}"
        sa = _collect_smoothed(runs_a, col)
        sb = _collect_smoothed(runs_b, col)
        ma, loa, hia, na = _series_mean_ci_per_step(sa)
        mb, lob, hib, nb = _series_mean_ci_per_step(sb)
        if ma.size == 0:
            continue
        x = np.arange(ma.size, dtype=np.int64)
        plt.figure(figsize=(10, 4))
        plt.plot(x, ma, label=f"All uncalibrated (n={na})", linewidth=1.8, color="C0")
        if na >= 2 and np.all(np.isfinite(loa)) and np.all(np.isfinite(hia)):
            plt.plot(x, loa, "--", color="C0", alpha=0.55, linewidth=1.0)
            plt.plot(x, hia, "--", color="C0", alpha=0.55, linewidth=1.0)
        plt.plot(x, mb, label=f"Gemma2 calibrated (n={nb})", linewidth=1.8, color="C1")
        if nb >= 2 and np.all(np.isfinite(lob)) and np.all(np.isfinite(hib)):
            plt.plot(x, lob, "--", color="C1", alpha=0.55, linewidth=1.0)
            plt.plot(x, hib, "--", color="C1", alpha=0.55, linewidth=1.0)
        plt.xlabel("Training batch")
        plt.ylabel("Mean wager")
        plt.title(f"{title_prefix} — {model_display_names[mi]}: mean wager ±95% CI (window={window})")
        plt.legend()
        plt.tight_layout()
        (out_dir / "plot_set_1").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "plot_set_1" / f"{title_prefix}__mean_wager__{model_display_names[mi]}.png", dpi=200)
        plt.close()

    # Plot set 2: net payout
    for mi in range(n_models):
        col = f"net_payout_model_{mi}"
        sa = _collect_smoothed(runs_a, col)
        sb = _collect_smoothed(runs_b, col)
        ma, loa, hia, na = _series_mean_ci_per_step(sa)
        mb, lob, hib, nb = _series_mean_ci_per_step(sb)
        if ma.size == 0:
            continue
        x = np.arange(ma.size, dtype=np.int64)
        plt.figure(figsize=(10, 4))
        plt.plot(x, ma, label=f"All uncalibrated (n={na})", linewidth=1.8, color="C0")
        if na >= 2:
            plt.plot(x, loa, "--", color="C0", alpha=0.55, linewidth=1.0)
            plt.plot(x, hia, "--", color="C0", alpha=0.55, linewidth=1.0)
        plt.plot(x, mb, label=f"Gemma2 calibrated (n={nb})", linewidth=1.8, color="C1")
        if nb >= 2:
            plt.plot(x, lob, "--", color="C1", alpha=0.55, linewidth=1.0)
            plt.plot(x, hib, "--", color="C1", alpha=0.55, linewidth=1.0)
        plt.xlabel("Training batch")
        plt.ylabel("Mean net payout")
        plt.title(f"{title_prefix} — {model_display_names[mi]}: mean net payout ±95% CI (window={window})")
        plt.legend()
        plt.tight_layout()
        (out_dir / "plot_set_2").mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "plot_set_2" / f"{title_prefix}__mean_net_payout__{model_display_names[mi]}.png", dpi=200)
        plt.close()


def _format_latex_mean_pm(
    mean: float,
    ci_low: Optional[float],
    ci_high: Optional[float],
    n: int,
    *,
    scale_100: bool,
) -> str:
    """Scaled mean (×100 when scale_100), 2 or 4 decimals, \\tiny{$\\pm$halfwidth} when CI known; mirrors repeat-script LaTeX style."""
    scale = 100.0 if scale_100 else 1.0
    dec = 2 if scale_100 else 4
    m = float(mean) * scale
    if ci_low is not None and ci_high is not None and n >= 2:
        lo_f = float(ci_low) * scale
        hi_f = float(ci_high) * scale
        margin = (hi_f - lo_f) / 2.0
        return f"{m:.{dec}f}\\tiny{{$\\pm${margin:.{dec}f}}}  $n={int(n)}$"
    return f"{m:.{dec}f}  $n={int(n)}$"


def print_test_bar_metrics_latex(
    *,
    title_prefix: str,
    eval_dataset_name: str,
    model_display_names: List[str],
    runs_unc: List[RunPaths],
    runs_cal: List[RunPaths],
    out_txt: Optional[Path] = None,
) -> str:
    """
    Print / save plot_set 3–5 summaries (per-model test means ±95%% CI) in LaTeX-friendly lines.
    All three metrics use ×100 and 2 decimals for consistency.
    """
    lines: List[str] = []
    lines.append(f"=== {title_prefix}: test bar metrics (LaTeX)  eval={eval_dataset_name} ===")
    blocks = [
        ("plot_set_3", "avg_wager_per_model", "Average normalized wager (ID test)", True),
        ("plot_set_4", "avg_net_payout_per_model", "Average net payout (ID test)", True),
        ("plot_set_5", "avg_sigmoid_wager_per_model", "Average unnormalized wager (sigmoid, ID test)", True),
    ]
    for plot_id, key, desc, scale_100 in blocks:
        mu, lou, hou, nu = _per_model_mean_ci_from_runs(runs_unc, eval_dataset_name, key)
        mc, loc, hoc, nc = _per_model_mean_ci_from_runs(runs_cal, eval_dataset_name, key)
        if mu.size == 0 or mc.size == 0:
            lines.append(f"# {plot_id} {desc}: (missing data)")
            lines.append("")
            continue
        lines.append(f"# {plot_id}: {desc}")
        lines.append("All uncalibrated:")
        for mi, name in enumerate(model_display_names):
            if mi >= mu.shape[0]:
                break
            lo_i = float(lou[mi]) if nu >= 2 and mi < lou.shape[0] and np.isfinite(lou[mi]) else None
            hi_i = float(hou[mi]) if nu >= 2 and mi < hou.shape[0] and np.isfinite(hou[mi]) else None
            lines.append(
                f"  {name}: {_format_latex_mean_pm(float(mu[mi]), lo_i, hi_i, nu, scale_100=scale_100)}"
            )
        lines.append("Gemma2 calibrated:")
        for mi, name in enumerate(model_display_names):
            if mi >= mc.shape[0]:
                break
            lo_i = float(loc[mi]) if nc >= 2 and mi < loc.shape[0] and np.isfinite(loc[mi]) else None
            hi_i = float(hoc[mi]) if nc >= 2 and mi < hoc.shape[0] and np.isfinite(hoc[mi]) else None
            lines.append(
                f"  {name}: {_format_latex_mean_pm(float(mc[mi]), lo_i, hi_i, nc, scale_100=scale_100)}"
            )
        lines.append("")
    text = "\n".join(lines)
    print(text, flush=True)
    if out_txt is not None:
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text)
    return text


def _per_model_mean_ci_from_runs(
    runs: List[RunPaths],
    eval_dataset_name: str,
    key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (mean_per_model, lo_per_model, hi_per_model, n_runs) for key in eval.results.json."""
    arrs: List[np.ndarray] = []
    for r in runs:
        d = _load_eval_dataset_result(r.eval_results_json, eval_dataset_name)
        v = d.get(key)
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float64).ravel()
        if a.size:
            arrs.append(a)
    if not arrs:
        return np.array([]), np.array([]), np.array([]), 0
    n_models = min(int(x.shape[0]) for x in arrs)
    mat = np.stack([x[:n_models] for x in arrs], axis=0)
    n = mat.shape[0]
    mean = np.mean(mat, axis=0)
    if n < 2:
        return mean, np.full(n_models, np.nan), np.full(n_models, np.nan), n
    from scipy import stats

    std = np.std(mat, axis=0, ddof=1)
    sem = std / float(np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    margin = tcrit * sem
    return mean, mean - margin, mean + margin, n


def _plot_test_bars_with_ci(
    *,
    out_dir: Path,
    title_prefix: str,
    model_display_names: List[str],
    eval_dataset_name: str,
    runs_a: List[RunPaths],
    runs_b: List[RunPaths],
) -> None:
    import matplotlib.pyplot as plt

    n_models = len(model_display_names)
    x = np.arange(n_models)
    width = 0.35

    def _bar_with_ci(
        key: str,
        ylabel: str,
        title: str,
        out_subdir: str,
        fname: str,
    ) -> None:
        ma, loa, hia, na = _per_model_mean_ci_from_runs(runs_a, eval_dataset_name, key)
        mb, lob, hib, nb = _per_model_mean_ci_from_runs(runs_b, eval_dataset_name, key)
        if ma.size != n_models or mb.size != n_models:
            raise ValueError(f"Mismatch per-model length for {key}: expected {n_models}")

        err_a = None
        err_b = None
        if na >= 2 and np.all(np.isfinite(loa)) and np.all(np.isfinite(hia)):
            err_a = np.vstack([ma - loa, hia - ma])
        if nb >= 2 and np.all(np.isfinite(lob)) and np.all(np.isfinite(hib)):
            err_b = np.vstack([mb - lob, hib - mb])

        fig, ax = plt.subplots(figsize=(10, 4))
        bars_a = ax.bar(
            x - width / 2,
            ma,
            width,
            yerr=err_a,
            capsize=4,
            label=f"All uncalibrated (n={na})",
            color="C0",
        )
        bars_b = ax.bar(
            x + width / 2,
            mb,
            width,
            yerr=err_b,
            capsize=4,
            label=f"Gemma2 calibrated (n={nb})",
            color="C1",
        )
        ax.set_xticks(x, model_display_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        for b in bars_a:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                float(b.get_height()),
                f"{float(b.get_height()):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for b in bars_b:
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                float(b.get_height()),
                f"{float(b.get_height()):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        plt.tight_layout()
        (out_dir / out_subdir).mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / out_subdir / fname, dpi=200)
        plt.close()

    _bar_with_ci(
        "avg_wager_per_model",
        "Average wager (normalized)",
        f"{title_prefix} — Average Normalized Wagers (mean ±95% CI)",
        "plot_set_3",
        f"{title_prefix}__avg_wager_test.png",
    )
    _bar_with_ci(
        "avg_net_payout_per_model",
        "Average net payout",
        f"{title_prefix} — Average Net Payout (mean ±95% CI)",
        "plot_set_4",
        f"{title_prefix}__avg_net_payout_test.png",
    )
    _bar_with_ci(
        "avg_sigmoid_wager_per_model",
        "Average wager (unnormalized)",
        f"{title_prefix} — Average Wagers (mean ±95% CI)",
        "plot_set_5",
        f"{title_prefix}__avg_unnormalized_wager_test.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmlu-config",
        type=str,
        default=str(_repo_root() / "examples" / "configs" / "wagering_training" / "mse_br_wagers_v2_4models_mmlu.yaml"),
    )
    parser.add_argument(
        "--medmcqa-config",
        type=str,
        default=str(_repo_root() / "examples" / "configs" / "wagering_training" / "mse_br_wagers_v2_4models_medmcqa.yaml"),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / "artifacts" / "calibration_plots"),
    )
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1)
    args = parser.parse_args()

    out_dir = _resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid importing repo code paths on module import.
    from wagering.utils import load_and_merge_configs

    for cfg_path_str, tag in [(args.mmlu_config, "MMLU"), (args.medmcqa_config, "MedMCQA")]:
        base_cfg_path = Path(cfg_path_str).expanduser().resolve()
        a_cfg, b_cfg = _make_setting_configs(base_cfg_path, num_epochs=int(args.num_epochs))

        merged_cfg = load_and_merge_configs(base_cfg_path)
        model_display_names = _display_names_for_models(merged_cfg)
        eval_ds_name = _infer_primary_test_display_name(merged_cfg)

        run_a = _run_pipeline(a_cfg)
        run_b = _run_pipeline(b_cfg)

        ds_out = out_dir / tag
        ds_out.mkdir(parents=True, exist_ok=True)

        _plot_training_curves(
            out_dir=ds_out,
            title_prefix=tag,
            model_display_names=model_display_names,
            run_a=run_a,
            run_b=run_b,
            window=int(args.smooth_window),
        )
        _plot_test_bars(
            out_dir=ds_out,
            title_prefix=tag,
            model_display_names=model_display_names,
            eval_dataset_name=eval_ds_name,
            run_a=run_a,
            run_b=run_b,
        )
        print_test_bar_metrics_latex(
            title_prefix=tag,
            eval_dataset_name=eval_ds_name,
            model_display_names=model_display_names,
            runs_unc=[run_a],
            runs_cal=[run_b],
            out_txt=ds_out / f"{tag}_plot_bars_latex_metrics.txt",
        )

        # Persist a short run manifest for reproducibility.
        manifest = {
            "dataset": tag,
            "base_config": str(base_cfg_path),
            "eval_dataset_name": eval_ds_name,
            "models": model_display_names,
            "runs": {
                "uncalibrated": {
                    "config": str(a_cfg),
                    "checkpoint_dir": str(run_a.checkpoint_dir),
                },
                "gemma2_calibrated": {
                    "config": str(b_cfg),
                    "checkpoint_dir": str(run_b.checkpoint_dir),
                },
            },
        }
        with (ds_out / "run_manifest.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    print(f"Wrote plots under: {out_dir}")


if __name__ == "__main__":
    main()

