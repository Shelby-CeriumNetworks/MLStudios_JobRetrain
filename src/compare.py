import os, json, shutil
from pathlib import Path
from typing import Any, Dict
from src.utils import append_jsonl, write_json, ensure_dir, now_ts

# ---- Paths (match your training code) ----
METRICS_DIR = "artifacts/metrics"
MODELS_DIR = "artifacts/models"

HISTORY_PATH = f"{METRICS_DIR}/metrics_history.jsonl"
LATEST_PATH  = f"{METRICS_DIR}/metrics_latest.json"

# Your training saves here:
TRAINED_MODEL_PATH = f"{MODELS_DIR}/logistic_regression_model.joblib"

# Canonical symlinks/copies for orchestration:
LATEST_MODEL_PATH = f"{MODELS_DIR}/model_latest.joblib"
BEST_MODEL_PATH   = f"{MODELS_DIR}/model_best.joblib"

# ---- Metric selection ----
# Prefer explicit env var; otherwise pick from common keys your code returns.
TARGET_METRIC = os.getenv("TARGET_METRIC")  # e.g. "roc_auc" or "auc" or "accuracy"
HIGHER_IS_BETTER = os.getenv("HIGHER_IS_BETTER", "true").lower() == "true"

def _to_builtin(v):
    if isinstance(v, np.generic):
        return v.item()
    return v

def _pick_metric_key(metrics: Dict[str, Any]) -> str | None:
    """
    Choose which scalar metric to compare.
    Priority: env TARGET_METRIC > 'roc_auc' > 'auc' > 'accuracy'
    Only returns keys that are numeric scalars (not lists).
    """
    import numbers
    if TARGET_METRIC and TARGET_METRIC in metrics and isinstance(metrics[TARGET_METRIC], numbers.Number):
        return TARGET_METRIC

    for k in ["roc_auc", "auc", "accuracy"]:
        if k in metrics and isinstance(metrics[k], numbers.Number):
            return k
    # no good scalar found
    return None

def _read_best_score(metric_key: str) -> float | None:
    if not os.path.exists(HISTORY_PATH):
        return None
    best = None
    with open(HISTORY_PATH, "r") as f:
        for line in f:
            rec = json.loads(line)
            m = rec.get("metrics", {})
            if metric_key not in m:
                continue
            score = m[metric_key]
            if not isinstance(score, (int, float)):
                continue
            if best is None:
                best = score
            else:
                best = max(best, score) if HIGHER_IS_BETTER else min(best, score)
    return best

def _snapshot_latest_model():
    """Make/refresh a 'latest' copy so promotion logic can work consistently."""
    if os.path.exists(TRAINED_MODEL_PATH):
        shutil.copyfile(TRAINED_MODEL_PATH, LATEST_MODEL_PATH)

def compare_and_promote(run_info: dict):
    """
    run_info shape (suggested):
      {
        "run_id": "...",
        "metrics": {...},           # from your train fn
        "meta": {...optional...}    # e.g. dataset source, random_state, etc.
      }
    """
    ensure_dir(METRICS_DIR)
    ensure_dir(MODELS_DIR)

    latest_metrics = run_info.get("metrics", {})
    write_json(LATEST_PATH, latest_metrics)

    # Append to history (store only non-huge values)
    record = {
        "ts": now_ts(),
        "run_id": run_info.get("run_id"),
        # keep full metrics (JSON) — but your lists can be large; OK if you want them
        "metrics": latest_metrics,
        "meta": run_info.get("meta", {}),
    }
    append_jsonl(HISTORY_PATH, record)

    metric_key = _pick_metric_key(latest_metrics)
    best_score = _read_best_score(metric_key) if metric_key else None
    latest_score = latest_metrics.get(metric_key) if metric_key else None

    improved = False
    if metric_key is not None and isinstance(latest_score, (int, float)):
        if best_score is None:
            improved = True
        else:
            improved = (latest_score > best_score) if HIGHER_IS_BETTER else (latest_score < best_score)

    # Always refresh "latest" pointer; then promote to "best" if improved
    _snapshot_latest_model()
    if improved and os.path.exists(LATEST_MODEL_PATH):
        shutil.copyfile(LATEST_MODEL_PATH, BEST_MODEL_PATH)

    return {
        "metric_key": metric_key,
        "improved": bool(improved),
        "best_score": _to_builtin(best_score) if best_score is not None else None,
        "latest_score": _to_builtin(latest_score) if latest_score is not None else None,
    }
