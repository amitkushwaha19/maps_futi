# futiplot/momentum/calibrate_percentiles.py
# ---------------------------------------------------------------------
# Percentile calibration utilities for chart elements.
#
# This module provides:
#
# MOMENTUM (per-bin “prevalue” magnitudes)
#   1) compute_momentum_prevalue_percentiles(df, bin_seconds)
#   2) save_momentum_prevalue_percentiles_csv(quantiles, out_path)
#   3) compute_and_save_momentum_percentiles_csv(df, out_path, bin_seconds)
#   4) load_momentum_prevalue_percentiles_cached(csv_path)  [LRU cached]
#   5) prevalue_to_percentile(values, quantiles)
#
# SHOT xG (for dot sizes on shot charts)
#   6) compute_shot_xg_percentiles(df)
#   7) save_shot_xg_percentiles_csv(quantiles, out_path)
#   8) compute_and_save_shot_xg_percentiles_csv(df, out_path)
#   9) load_shot_xg_percentiles_cached(csv_path)            [LRU cached]
#  10) xg_to_percentile(xg_values, quantiles)
#  11) xg_to_point_size(xg_values, quantiles, size_min, size_max)
#
# Typical offline workflow:
#   q_mom = compute_momentum_prevalue_percentiles(history_df, bin_seconds=120)
#   save_momentum_prevalue_percentiles_csv(q_mom, "momentum_prevalue_percentiles_120.csv")
#
#   q_xg = compute_shot_xg_percentiles(history_df)
#   save_shot_xg_percentiles_csv(q_xg, "shots_xg_percentiles.csv")
#
# Typical runtime workflow:
#   q_mom = load_momentum_prevalue_percentiles_cached("momentum_prevalue_percentiles_120.csv")
#   p_mom = prevalue_to_percentile(per_bin_prevalues, q_mom)  # → [0..1]
#
#   q_xg  = load_shot_xg_percentiles_cached("shots_xg_percentiles.csv")
#   sizes = xg_to_point_size(match_shots["xg"].to_numpy(), q_xg, size_min=6, size_max=22)
# ---------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    # momentum calibration
    "compute_momentum_prevalue_percentiles",
    "save_momentum_prevalue_percentiles_csv",
    "compute_and_save_momentum_percentiles_csv",
    "load_momentum_prevalue_percentiles_cached",
    "prevalue_to_percentile",
    # shot xg calibration
    "compute_shot_xg_percentiles",
    "save_shot_xg_percentiles_csv",
    "compute_and_save_shot_xg_percentiles_csv",
    "load_shot_xg_percentiles_cached",
    "xg_to_percentile",
    "xg_to_point_size",
]

# =====================================================================
# Shared helpers
# =====================================================================

_PGRID = np.linspace(0.0, 1.0, 101)  # 0..100% at 1% steps


def _safe_quantile(values: np.ndarray) -> np.ndarray:
    """
    Defensive percentile computation on a 1D float array.
    Returns a (101,) vector of the 0..100 percentiles (linear method).
    """
    if values.size == 0:
        return np.zeros(101, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(101, dtype=float)
    return np.quantile(values, _PGRID, method="linear").astype(float, copy=False)


def _invert_with_quantiles(x: np.ndarray, qvec: np.ndarray) -> np.ndarray:
    """
    Given raw values x and a (101,) quantile vector qvec (value at each percentile),
    return percentiles in [0..1] by linear interpolation and clamping.
    """
    if qvec is None or qvec.shape != (101,) or not np.isfinite(qvec).any():
        return np.zeros_like(x, dtype=float)
    # enforce non-decreasing for numerical stability
    q = np.maximum.accumulate(np.asarray(qvec, dtype=float))
    x = np.where(np.isfinite(x), x, 0.0)
    return np.interp(x, q, _PGRID, left=0.0, right=1.0).astype(float, copy=False)


# =====================================================================
# Momentum calibration (per-bin abs(max prevalue))
# =====================================================================

def compute_momentum_prevalue_percentiles(
    df: pd.DataFrame,
    bin_seconds: int = 120,
) -> np.ndarray:
    """
    Compute a global calibration vector for the momentum plot.

    Logic:
      - Keep possession actions only: pass, dribble, carry, reception, shot.
      - prevalue = prev_scores - prev_concedes (per event).
      - Bin by (match_id, period_id) using the observed period start (min time_seconds).
      - Aggregate by (match_id, team_id, period_id, time_bin).
      - Take abs(max(prevalue)) within each team/bin (mirrors bar-height symmetry).
      - Return 0..100 percentiles of those per-bin magnitudes.

    Inputs:
      df must contain:
        ["match_id","team_id","period_id","time_seconds","type_name",
         "prev_scores","prev_concedes"]

    Returns:
      np.ndarray shape (101,), where out[q] is the qth percentile.
      Returns zeros if input is empty/invalid.
    """
    need = {
        "match_id", "team_id", "period_id", "time_seconds",
        "type_name", "prev_scores", "prev_concedes",
    }
    if df.empty or not need.issubset(df.columns):
        return np.zeros(101, dtype=float)

    keep_actions = {"pass", "dribble", "carry", "reception", "shot"}
    x = (
        df.loc[df["type_name"].isin(keep_actions),
               ["match_id", "team_id", "period_id", "time_seconds", "prev_scores", "prev_concedes"]]
          .dropna(subset=["period_id", "time_seconds", "prev_scores", "prev_concedes"])
          .copy()
    )
    if x.empty:
        return np.zeros(101, dtype=float)

    # within-period bin index using observed period start
    t0 = x.groupby(["match_id", "period_id"])["time_seconds"].transform("min")
    x["time_bin"] = ((x["time_seconds"] - t0) // bin_seconds).astype(int)

    # per-bin abs(max prevalue)
    x["prevalue"] = x["prev_scores"] - x["prev_concedes"]
    per_bin = (
        x.groupby(["match_id", "team_id", "period_id", "time_bin"], observed=True)["prevalue"]
         .max()
         .abs()
         .to_numpy()
         .astype(float, copy=False)
    )
    return _safe_quantile(per_bin)


def save_momentum_prevalue_percentiles_csv(quantiles: np.ndarray, out_path: str) -> None:
    """
    Write a CSV with two columns: q,value where q=0..100 (ints), value=float percentile.
    """
    q = np.arange(101, dtype=int)
    pd.DataFrame({"q": q, "value": np.asarray(quantiles, dtype=float)}).to_csv(out_path, index=False)


def compute_and_save_momentum_percentiles_csv(
    df: pd.DataFrame,
    out_path: str,
    bin_seconds: int = 120,
) -> np.ndarray:
    """
    Convenience wrapper: compute then save, returning the quantile vector.
    """
    quantiles = compute_momentum_prevalue_percentiles(df, bin_seconds=bin_seconds)
    save_momentum_prevalue_percentiles_csv(quantiles, out_path)
    return quantiles


@lru_cache(maxsize=8)
def load_momentum_prevalue_percentiles_cached(csv_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load the momentum 0..100 percentile vector from CSV and cache via LRU.

    CSV format:
      q,value
      0,0.00000
      ...
      100,0.98765

    Args:
      csv_path: Path to CSV. If None or read fails, returns None.

    Returns:
      np.ndarray (101,) or None on failure.
    """
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["value"])
        vec = df["value"].to_numpy(dtype=float)
        if vec.shape[0] == 101 and np.isfinite(vec).any():
            return vec
    except Exception:
        pass
    return None


def prevalue_to_percentile(
    values: np.ndarray | list[float] | pd.Series,
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Map raw per-bin prevalue magnitudes to percentiles in [0..1] using the
    precomputed momentum quantile vector.
    """
    x = np.asarray(values, dtype=float)
    return _invert_with_quantiles(x, quantiles)


# =====================================================================
# Shot xG calibration (for dot sizes)
# =====================================================================

def compute_shot_xg_percentiles(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a global calibration vector for shot xG.

    Logic:
      - Keep rows where type_name == "shot".
      - Drop rows with missing/invalid xG.
      - Return 0..100 percentiles of the xG distribution.

    Inputs:
      df must contain at least: ["type_name", "xg"].

    Returns:
      np.ndarray shape (101,), where out[q] is the qth percentile value.
      Returns zeros if input is empty/invalid.
    """
    if df.empty or ("xg" not in df.columns):
        return np.zeros(101, dtype=float)

    shots = (
        df.loc[df.get("type_name", "") == "shot", ["xg"]]
          .dropna(subset=["xg"])
          .copy()
    )
    if shots.empty:
        return np.zeros(101, dtype=float)

    xg = pd.to_numeric(shots["xg"], errors="coerce").to_numpy(dtype=float)
    xg = xg[np.isfinite(xg)]
    return _safe_quantile(xg)


def save_shot_xg_percentiles_csv(quantiles: np.ndarray, out_path: str) -> None:
    """
    Write a CSV with two columns: q,value where q=0..100 (ints), value=float percentile.
    """
    q = np.arange(101, dtype=int)
    pd.DataFrame({"q": q, "value": np.asarray(quantiles, dtype=float)}).to_csv(out_path, index=False)


def compute_and_save_shot_xg_percentiles_csv(df: pd.DataFrame, out_path: str) -> np.ndarray:
    """
    Convenience wrapper: compute quantiles and save them to CSV. Returns the quantile vector.
    """
    q = compute_shot_xg_percentiles(df)
    save_shot_xg_percentiles_csv(q, out_path)
    return q


@lru_cache(maxsize=8)
def load_shot_xg_percentiles_cached(csv_path: Optional[str]) -> Optional[np.ndarray]:
    """
    Load the shot xG 0..100 percentile vector from CSV and cache via LRU.

    Args:
      csv_path: Path to CSV. If None or read fails, returns None.

    Returns:
      np.ndarray (101,) or None on failure.
    """
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path, usecols=["value"])
        vec = df["value"].to_numpy(dtype=float)
        if vec.shape[0] == 101 and np.isfinite(vec).any():
            return vec
    except Exception:
        pass
    return None


def xg_to_percentile(
    xg_values: np.ndarray | list[float] | pd.Series,
    quantiles: np.ndarray,
) -> np.ndarray:
    """
    Map raw shot xG values to percentiles in [0..1] using a precomputed quantile vector.
    """
    x = np.asarray(xg_values, dtype=float)
    return _invert_with_quantiles(x, quantiles)


def xg_to_point_size(
    xg_values: np.ndarray | list[float] | pd.Series,
    quantiles: np.ndarray,
    *,
    size_min: float = 6.0,
    size_max: float = 22.0,
) -> np.ndarray:
    """
    Convert raw xG values to point sizes for plotting, via percentile scaling.

    Steps:
      1) xG → percentile in [0..1] with xg_to_percentile.
      2) Linear map percentile into [size_min..size_max].

    Args:
      xg_values: Iterable of raw xG values for the current match.
      quantiles: Global 0..100 percentile vector for historical xG.
      size_min: Pixel size at the 0th percentile (inclusive).
      size_max: Pixel size at the 100th percentile (inclusive).

    Returns:
      np.ndarray of float sizes, same length/order as xg_values.
    """
    p = xg_to_percentile(xg_values, quantiles)
    p = np.clip(p, 0.0, 1.0)
    sizes = size_min + p * (size_max - size_min)
    return sizes.astype(float, copy=False)


# =====================================================================
# Heatmap bin scalers (percentile-based: value → alpha in [0,1])
# =====================================================================

from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# keep the same public API names
try:
    __all__
except NameError:
    __all__ = []

__all__ += [
    "fit_bin_scaler",
    "apply_bin_scaler",
    "save_bin_scaler_csv",
    "compute_and_save_bin_scaler_csv",
    "load_bin_scaler_cached",
]


def _percentiles_0_100(values: np.ndarray) -> np.ndarray:
    """
    Compute percentiles at q=0..100 (101 points), enforcing monotone non-decreasing.
    Uses numpy's modern 'method' arg, with a fallback to the older 'interpolation' arg.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        qvec = np.zeros(101, dtype=float)
    else:
        try:
            qvec = np.percentile(v, np.arange(101), method="linear").astype(float)
        except TypeError:
            # numpy < 1.22 fallback
            qvec = np.percentile(v, np.arange(101), interpolation="linear").astype(float)
    # be safe: enforce monotonicity
    qvec = np.maximum.accumulate(qvec)
    return qvec


def fit_bin_scaler(
    values: np.ndarray | list[float] | pd.Series,
) -> Dict[str, Any]:
    """
    Fit a simple percentile-based scaler on a 1D array of historical intensities
    (e.g., blurred per-90 counts). The scaler stores the thresholds at q=0..100.

    Returns dict:
      {"method": "percentile", "version": 1, "qvec": np.ndarray shape (101,)}
    """
    x = np.asarray(values, dtype=float)
    # Clamp negatives to 0 for safety — intensities should be ≥ 0
    x = np.where(np.isfinite(x) & (x >= 0.0), x, 0.0)
    qvec = _percentiles_0_100(x)
    return {"method": "percentile", "version": 1, "qvec": qvec}


def apply_bin_scaler(
    values: np.ndarray | list[float] | pd.Series,
    scaler: Dict[str, Any] | None,
) -> np.ndarray:
    """
    Map values → alpha in [0,1] using the fitted percentile thresholds.

    For each x:
      alpha = interp(x, qvec, pgrid) with pgrid = linspace(0,1,101)

    If scaler is None/invalid: returns zeros.
    """
    x = np.asarray(values, dtype=float)
    x = np.where(np.isfinite(x) & (x >= 0.0), x, 0.0)

    if not isinstance(scaler, dict) or scaler.get("method") != "percentile":
        return np.zeros_like(x, dtype=float)

    qvec = np.asarray(scaler.get("qvec", None), dtype=float)
    if qvec.shape != (101,) or not np.all(np.isfinite(qvec)):
        return np.zeros_like(x, dtype=float)

    # ensure monotone thresholds
    qvec = np.maximum.accumulate(qvec)
    pgrid = np.linspace(0.0, 1.0, 101)

    out = np.interp(x, qvec, pgrid, left=0.0, right=1.0)
    # already in [0,1] from interp; clip anyway for numerical safety
    return np.clip(out, 0.0, 1.0).astype(float, copy=False)


def save_bin_scaler_csv(scaler: Dict[str, Any], out_path: str | Path) -> None:
    """
    Persist a percentile scaler to CSV with two columns: q,value (q=0..100).
    """
    if not isinstance(scaler, dict) or scaler.get("method") != "percentile":
        raise ValueError("save_bin_scaler_csv expects a percentile scaler dict.")

    qvec = np.asarray(scaler["qvec"], dtype=float)
    rows = pd.DataFrame({"q": np.arange(101, dtype=int), "value": qvec})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_path, index=False)


def compute_and_save_bin_scaler_csv(
    values: np.ndarray | list[float] | pd.Series,
    out_path: str | Path,
) -> Dict[str, Any]:
    """
    Fit a percentile scaler on values and save it to CSV (q,value).
    Returns the scaler dict.
    """
    scaler = fit_bin_scaler(values)
    save_bin_scaler_csv(scaler, out_path)
    return scaler


@lru_cache(maxsize=16)
def load_bin_scaler_cached(csv_path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    """
    Load a percentile scaler from CSV and cache via LRU.
    Expects two columns: q,value for q=0..100.
    Returns {"method":"percentile", "version":1, "qvec": ...} or None on failure.
    """
    if not csv_path:
        return None
    p = Path(csv_path)
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)
        # Accept either exact headers 'q'/'value' or the first two numeric columns.
        if {"q", "value"}.issubset(df.columns):
            vals = df["value"].to_numpy(dtype=float)
        else:
            # fallback: take 2nd column as values if no headers match
            vals = df.iloc[:, 1].to_numpy(dtype=float)
        vals = np.asarray(vals, dtype=float)
        if vals.size < 101:
            return None
        qvec = vals[:101]
        if not np.isfinite(qvec).all():
            return None
        qvec = np.maximum.accumulate(qvec)
        return {"method": "percentile", "version": 1, "qvec": qvec}
    except Exception:
        return None

