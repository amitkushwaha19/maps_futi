# match_card_charts.py
# ---------------------------------------------------------------------
# compact, well-documented transforms for momentum, xG step, territory,
# shotmap, and action heatmap. optimized for repeated, per-minute runs.
# ---------------------------------------------------------------------

#%%
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from futiplot.utils import futicolor, transform_xy

from pathlib import Path
from futiplot.match_card.calibrate_percentiles import load_bin_scaler_cached, apply_bin_scaler

# …/src/futiplot/match_card/match_card_charts.py → …/src/futiplot/data
_PKG_DATA_DIR = (Path(__file__).resolve().parent.parent / "data")

#%%
# =====================================================================
# timing meta (shared between charts)
# =====================================================================

def compute_timing_meta(
    df: pd.DataFrame,
    *,
    bin_seconds: int = 120,
    plot_gap_seconds: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-game timing metadata and observed period bounds.

    Returns:
      meta: dataframe indexed by match_id with:
        - L1_obs..L4_obs: observed durations per period
        - latest_period: last observed period number
        - len_p1..len_p4: assumed lengths per period
        - off_p1..off_p4: stitching offsets from observed ends
        - D: total assumed duration (seconds) used to scale x=0..100
        - if plot_gap_seconds is provided:
            * D_plot: D plus gaps inserted after finished periods
            * off_p2_plot/off_p3_plot/off_p4_plot: observed offsets plus plot gaps
      per_bounds: dataframe with per-(game, period) observed [t0,t1]
    """
    # keep only what's needed to build bounds
    need = {"match_id", "period_id", "time_seconds"}
    if df.empty or not need.issubset(df.columns):
        # empty shells with expected columns so callers can join safely
        meta = pd.DataFrame(
            columns=["match_id", "L1_obs", "L2_obs", "L3_obs", "L4_obs",
                     "latest_period", "len_p1", "len_p2", "len_p3", "len_p4",
                     "off_p1", "off_p2", "off_p3", "off_p4", "D"]
        )
        per_bounds = pd.DataFrame(columns=["match_id", "period_id", "t0", "t1"])
        return meta, per_bounds

    # observed period bounds
    bounds = (
        df[["match_id", "period_id", "time_seconds"]]
        .dropna(subset=["period_id", "time_seconds"])
        .copy()
    )
    bounds["period_id"] = pd.to_numeric(bounds["period_id"], errors="coerce").astype(int)

    per_bounds = (
        bounds.groupby(["match_id", "period_id"], observed=True, sort=False)["time_seconds"]
              .agg(t0="min", t1="max")
              .reset_index()
    )

    # observed durations per period → wide by game
    dur = per_bounds.assign(L_obs=(per_bounds["t1"] - per_bounds["t0"]).astype(float))
    L = dur.pivot(index="match_id", columns="period_id", values="L_obs")
    if not L.empty:
        L.columns = pd.to_numeric(L.columns, errors="coerce").astype(int)
    L = (
        L.reindex(columns=range(1, 5), fill_value=0.0)
         .rename(columns={1: "L1_obs", 2: "L2_obs", 3: "L3_obs", 4: "L4_obs"})
         [["L1_obs", "L2_obs", "L3_obs", "L4_obs"]]
    )

    latest = (
        per_bounds.groupby("match_id", observed=True, sort=False)["period_id"]
                  .max()
                  .rename("latest_period")
    )

    # assumed period lengths (sec): halves 45', extra-time halves 15'
    H, E = 45 * 60.0, 15 * 60.0
    meta = pd.concat([L, latest], axis=1).reset_index().fillna(0.0)
    lp = meta["latest_period"].to_numpy()

    for k, default_k in ((1, H), (2, H), (3, E), (4, E)):
        obs = meta[f"L{k}_obs"].to_numpy()
        meta[f"len_p{k}"] = np.where(
            lp > k,  # finished
            np.where(obs > 0.0, obs, default_k),
            np.where(      # current
                lp == k,
                np.where(obs <= default_k, default_k, obs + 60.0),
                np.where((k == 2) & (lp == 1), H, 0.0),  # future
            ),
        )

    # observed stitching offsets and total assumed duration D
    meta["off_p1"] = 0.0
    meta["off_p2"] = meta["L1_obs"]
    meta["off_p3"] = meta["L1_obs"] + meta["L2_obs"]
    meta["off_p4"] = meta["L1_obs"] + meta["L2_obs"] + meta["L3_obs"]
    meta["D"] = meta[["len_p1", "len_p2", "len_p3", "len_p4"]].sum(axis=1)

    # optional: plot-time augmentation with gaps after finished periods
    if plot_gap_seconds is not None:
        gap_s = float(plot_gap_seconds)
        n_breaks = np.clip((lp - 1).astype(int), 0, 3)
        meta["D_plot"] = meta["D"] + n_breaks * gap_s

        gap_before_p2 = gap_s * np.minimum(1, n_breaks)
        gap_before_p3 = gap_s * np.minimum(2, n_breaks)
        gap_before_p4 = gap_s * np.minimum(3, n_breaks)

        meta["off_p1_plot"] = 0.0
        meta["off_p2_plot"] = meta["off_p2"] + gap_before_p2
        meta["off_p3_plot"] = meta["off_p3"] + gap_before_p3
        meta["off_p4_plot"] = meta["off_p4"] + gap_before_p4

    return meta, per_bounds


# =====================================================================
# cached percentile loader (used by momentum)
# =====================================================================

@lru_cache(maxsize=16)
def _load_momentum_percentiles_cached(bin_seconds: int, explicit_path: str | None) -> Optional[np.ndarray]:
    """
    Load the 0..100 percentile calibration vector for a given bin size.
    Uses LRU cache so repeated calls (per minute) avoid re-reading the CSV.
    """
    # explicit path (most specific) → fallback to package resource
    if explicit_path:
        try:
            arr = np.loadtxt(explicit_path, delimiter=",", skiprows=1)
            vec = (arr[:, 1] if arr.ndim == 2 else np.array([arr[1]])).astype(float)
            return vec if vec.size == 101 else None
        except Exception:
            pass

    try:
        p = _PKG_DATA_DIR / f"momentum_prevalue_percentiles_{bin_seconds}.csv"
        if p.exists():
            arr = np.loadtxt(p, delimiter=",", skiprows=1)
            vec = (arr[:, 1] if arr.ndim == 2 else np.array([arr[1]])).astype(float)
            return vec if vec.size == 101 else None
    except Exception:
        pass

    # nothing found
    return None
    
# =====================================================================
# cached percentile loader (for shot xG → marker sizing)
# =====================================================================

@lru_cache(maxsize=8)
def _load_shots_xg_percentiles_cached(explicit_path: str | None) -> Optional[np.ndarray]:
    """
    Load the 0..100 percentile vector for shot xG sizing.

    If `explicit_path` is provided, try that first (CSV with two columns: q,value).
    Otherwise, load from futiplot/data adjacent to this module.
    Returns:
        np.ndarray of shape (101,) with values increasing from q=0..100, or None on failure.
    """
    # 1) explicit file path, if provided
    if explicit_path:
        try:
            arr = np.loadtxt(explicit_path, delimiter=",", skiprows=1)
            vec = (arr[:, 1] if arr.ndim == 2 else np.array([arr[1]])).astype(float)
            return vec if vec.size == 101 else None
        except Exception:
            pass

    # 2) packaged default
    try:
        p = _PKG_DATA_DIR / "shots_xg_percentiles.csv"
        if p.exists():
            arr = np.loadtxt(p, delimiter=",", skiprows=1)
            vec = (arr[:, 1] if arr.ndim == 2 else np.array([arr[1]])).astype(float)
            return vec if vec.size == 101 else None
    except Exception:
        pass

    # nothing found
    return None

# =====================================================================
# cached percentile loader (for heatmaps)
# =====================================================================

@lru_cache(maxsize=8)
def _load_heatmap_scaler_both_cached(
    *,
    length_edges: int = 21,
    width_edges: int = 14,
    L: float = 105.0,
    W: float = 68.0,
    sigma: float | None = None,
    explicit_path: str | None = None,
) -> Optional[dict]:
    nx = int(length_edges) - 1
    ny = int(width_edges) - 1
    sigma_used = float(sigma) if sigma is not None else max(1.0, min(nx, ny) / 10.0)

    # 1) explicit path
    if explicit_path:
        sc = load_bin_scaler_cached(explicit_path)
        if sc:
            return sc

    # 2) futiplot/data next to this file
    fname = f"heatmap_scaler_both_L{int(L)}_W{int(W)}_Z{int(length_edges)}x{int(width_edges)}_sigma{sigma_used:.3g}.csv"
    sc = load_bin_scaler_cached(str(_PKG_DATA_DIR / fname))
    if sc:
        return sc

    # 3) packaged default (if bundled)
    try:
        import importlib.resources as ir
        p = ir.files("futiplot.data") / fname
        return load_bin_scaler_cached(str(p))
    except Exception:
        return None

@lru_cache(maxsize=8)
def _load_heatmap_scaler_team_cached(
    *,
    length_edges: int = 21,
    width_edges: int = 14,
    L: float = 105.0,
    W: float = 68.0,
    sigma: float | None = None,
    explicit_path: str | None = None,
) -> Optional[dict]:
    nx = int(length_edges) - 1
    ny = int(width_edges) - 1
    sigma_used = float(sigma) if sigma is not None else max(1.0, min(nx, ny) / 10.0)

    # 1) explicit path
    if explicit_path:
        sc = load_bin_scaler_cached(explicit_path)
        if sc:
            return sc

    # 2) futiplot/data next to this file
    fname = f"heatmap_scaler_team_L{int(L)}_W{int(W)}_Z{int(length_edges)}x{int(width_edges)}_sigma{sigma_used:.3g}.csv"
    sc = load_bin_scaler_cached(str(_PKG_DATA_DIR / fname))
    if sc:
        return sc

    # 3) packaged default (if bundled)
    try:
        import importlib.resources as ir
        p = ir.files("futiplot.data") / fname
        return load_bin_scaler_cached(str(p))
    except Exception:
        return None


# =====================================================================
# momentum
# =====================================================================

def df_match_momentum(
    df: pd.DataFrame,
    *,
    bin_seconds: int = 120,
    percentiles_path: str | None = None,
    pctl_floor: float = 0.0,
    smoothing: float = 1.0,
    width_scale: float = 1.0,
    meta: Optional[pd.DataFrame] = None,          # optional precomputed meta with plot gaps
    per_bounds: Optional[pd.DataFrame] = None,    # provided for api symmetry only
) -> pd.DataFrame:
    """
    build a momentum dataframe with percentile scaling, optional smoothing, and dynamic x
    (with partial-bin gaps between periods), plus minute ticks and special 'period_end' ticks

    output rows:
      type='bar'     → base bars (both teams), faded color, drawn from 50 outward
      type='bar_top' → overlay bar (winner only), solid color, difference from 50
      type='tick'    → vertical minute markers + special 'period_end' markers

    columns:
      type, match_id, team_id, home, period_id, time_bin, x0, x1, y0, y1,
      bar_color, bar_fill, pctl, x, label
    """
    # required columns
    need = {
        "match_id", "team_id", "home", "period_id", "time_seconds",
        "type_name", "prev_scores", "prev_concedes",
    }
    out_cols = [
        "type", "match_id", "team_id", "home", "period_id", "time_bin",
        "x0", "x1", "y0", "y1", "bar_color", "bar_fill", "pctl", "x", "label",
    ]
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=out_cols)

    # keep possession actions and bin within period
    keep_actions = {"pass", "dribble", "carry", "reception", "shot"}
    x = (
        df.loc[df["type_name"].isin(keep_actions),
               ["match_id", "team_id", "home", "period_id", "time_seconds",
                "prev_scores", "prev_concedes"]]
          .dropna(subset=["period_id", "time_seconds", "prev_scores", "prev_concedes"])
          .copy()
    )
    if x.empty:
        return pd.DataFrame(columns=out_cols)

    t0 = x.groupby(["match_id", "period_id"], observed=True, sort=False)["time_seconds"].transform("min")
    x["time_bin"] = ((x["time_seconds"] - t0) // bin_seconds).astype(int)

    # timing meta with plot gaps sized to 0.7 × bin_seconds
    if meta is None:
        meta, _ = compute_timing_meta(df, bin_seconds=bin_seconds, plot_gap_seconds=0.7 * bin_seconds)

    # per-team, per-bin absolute max prevalue
    x["prevalue"] = x["prev_scores"] - x["prev_concedes"]
    per_bin = (
        x.groupby(["match_id", "team_id", "home", "period_id", "time_bin"], observed=True, sort=False)["prevalue"]
         .max()
         .abs()
         .reset_index(name="pre_bin_absmax")
    )
    if per_bin.empty:
        return pd.DataFrame(columns=out_cols)

    # optional smoothing across adjacent bins within the same period
    s = float(max(0.0, smoothing))
    if s > 0.0:
        def _smooth_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("time_bin", kind="stable").copy()
            cur = g["pre_bin_absmax"].astype(float)
            tb  = g["time_bin"].astype(int)

            prev_val = cur.shift(1); next_val = cur.shift(-1)
            prev_ok = (tb - tb.shift(1) == 1); next_ok = (tb.shift(-1) - tb == 1)
            prev_val = prev_val.where(prev_ok, np.nan)
            next_val = next_val.where(next_ok, np.nan)

            num = cur + s * prev_val.fillna(0.0) + s * next_val.fillna(0.0)
            den = 1.0 + s * prev_val.notna().astype(float) + s * next_val.notna().astype(float)
            g["pre_bin_absmax_smooth"] = (num / den).astype(float)
            return g

        per_bin = (
            per_bin.groupby(["match_id", "team_id", "home", "period_id"], observed=True, sort=False, group_keys=False)
                   .apply(_smooth_group)
        )
        v_source = "pre_bin_absmax_smooth"
    else:
        v_source = "pre_bin_absmax"

    # map magnitudes to percentiles, with floor
    qvec = _load_momentum_percentiles_cached(bin_seconds, percentiles_path)
    v = per_bin[v_source].to_numpy(dtype=float)

    if qvec is not None and np.isfinite(qvec).all() and qvec.max() > qvec.min():
        qvec = np.maximum.accumulate(qvec)  # monotone
        pgrid = np.linspace(0.0, 1.0, 101)
        per_bin["pctl"] = np.clip(np.interp(v, qvec, pgrid, left=0.0, right=1.0), 0.0, 1.0)
    else:
        # in-match ecdf fallback
        if not np.isfinite(v).any() or np.nanmax(v) <= 0.0:
            per_bin["pctl"] = 0.0
        else:
            sv = np.sort(v[np.isfinite(v)])
            n = float(len(sv))
            per_bin["pctl"] = (np.searchsorted(sv, v, side="right") / n).clip(0.0, 1.0)

    f = float(pctl_floor)
    if f > 0.0:
        p = per_bin["pctl"].to_numpy(dtype=float)
        p = np.where(p <= f, 0.0, (p - f) / (1.0 - f))
        per_bin["pctl"] = np.clip(p, 0.0, 1.0)

    # plot-time bin edges for (game, period, time_bin) with width scaling
    bins_unique = per_bin[["match_id", "period_id", "time_bin"]].drop_duplicates().copy()
    bins_unique = bins_unique.merge(
        meta[["match_id", "D_plot", "off_p2_plot", "off_p3_plot", "off_p4_plot"]],
        on="match_id", how="left"
    )
    bins_unique["period_id"] = pd.to_numeric(bins_unique["period_id"], errors="coerce").astype(int)
    bins_unique["time_bin"]  = pd.to_numeric(bins_unique["time_bin"],  errors="coerce").astype(int)

    pid = bins_unique["period_id"].to_numpy()
    off_plot = np.where(pid == 2, bins_unique["off_p2_plot"].to_numpy(),
               np.where(pid == 3, bins_unique["off_p3_plot"].to_numpy(),
               np.where(pid == 4, bins_unique["off_p4_plot"].to_numpy(), 0.0)))
    t_abs_left  = bins_unique["time_bin"].to_numpy(dtype=float) * float(bin_seconds) + off_plot
    t_abs_right = t_abs_left + float(bin_seconds)

    D_plot = bins_unique["D_plot"].to_numpy(dtype=float)
    x0_full = np.where(D_plot > 0, 100.0 * t_abs_left  / D_plot, 0.0)
    x1_full = np.where(D_plot > 0, 100.0 * np.minimum(t_abs_right, D_plot) / D_plot, 0.0)

    ws = float(width_scale)
    width = (x1_full - x0_full) * ws
    center = 0.5 * (x0_full + x1_full)
    bins_unique["x0"] = np.clip(center - 0.5 * width, 0.0, 100.0)
    bins_unique["x1"] = np.clip(center + 0.5 * width, 0.0, 100.0)

    per_bin = per_bin.merge(
        bins_unique[["match_id", "period_id", "time_bin", "x0", "x1"]],
        on=["match_id", "period_id", "time_bin"], how="left"
    )

    # base bars (both teams), faded color
    base = per_bin.copy()
    base["type"] = "bar"
    dy = base["pctl"].to_numpy(dtype=float) * 50.0
    is_home = base["home"].astype(str).eq("home").to_numpy()

    base["y0"] = 50.0
    base["y1"] = np.where(is_home, 50.0 + dy, 50.0 - dy)
    base["bar_color"] = np.where(base["home"].eq("home"), futicolor.blue1, futicolor.pink1)
    base["bar_fill"] = np.nan
    base["x"] = np.nan
    base["label"] = np.nan
    base_out = base[out_cols].copy()

    # overlay bars (winner only), solid color
    key = ["match_id", "period_id", "time_bin"]
    wide = (
        per_bin.pivot(index=key, columns="home", values="pctl")
              .fillna(0.0)
              .reset_index()
    )
    wide = wide.merge(bins_unique[["match_id", "period_id", "time_bin", "x0", "x1"]], on=key, how="left")

    side_team = (
        df[["match_id", "home", "team_id"]].dropna()
          .drop_duplicates(["match_id", "home"])
          .pivot(index="match_id", columns="home", values="team_id")
    )
    def _map_team(ids: pd.Series, side: str) -> np.ndarray:
        if side in side_team.columns:
            return ids.map(side_team[side]).to_numpy()
        return np.full(len(ids), np.nan)

    p_home = (wide["home"] if "home" in wide.columns else 0.0).astype(float).to_numpy()
    p_away = (wide["away"] if "away" in wide.columns else 0.0).astype(float).to_numpy()
    diff = np.abs(p_home - p_away)
    home_wins = p_home >= p_away
    dy_diff = diff * 50.0

    y0_top = np.full_like(dy_diff, 50.0, dtype=float)
    y1_top = np.where(home_wins, 50.0 + dy_diff, 50.0 - dy_diff)

    top = pd.DataFrame({
        "type": "bar_top",
        "match_id": wide["match_id"],
        "team_id": np.where(home_wins, _map_team(wide["match_id"], "home"), _map_team(wide["match_id"], "away")),
        "home": np.where(home_wins, "home", "away"),
        "period_id": wide["period_id"],
        "time_bin": wide["time_bin"],
        "x0": wide["x0"],
        "x1": wide["x1"],
        "y0": y0_top,
        "y1": y1_top,
        "bar_color": np.nan,  # overlay uses bar_fill only
        "bar_fill": np.where(home_wins, futicolor.blue, futicolor.pink),
        "pctl": np.where(home_wins, p_home, p_away),
        "x": np.nan,
        "label": np.nan,
    })
    top = top.loc[(np.abs(top["y1"] - top["y0"]) > 1e-9)]

    # minute ticks in plot time
    minutes = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135)

    def _minute_abs_plot(m: int, off2: float, off3: float, off4: float) -> float | None:
        if m <= 45:   return m * 60.0
        if m <= 90:   return off2 + (m - 45) * 60.0
        if m <= 105:  return off3 + (m - 90) * 60.0
        if m <= 120:  return off4 + (m - 105) * 60.0
        return None

    meta_ticks = meta[["match_id", "D_plot", "off_p2_plot", "off_p3_plot", "off_p4_plot", "latest_period"]].copy()

    tick_rows = []
    for row in meta_ticks.itertuples(index=False):
        if row.D_plot <= 0:
            continue
        for m in minutes:
            t_plot = _minute_abs_plot(m, row.off_p2_plot, row.off_p3_plot, row.off_p4_plot)
            if t_plot is not None and t_plot <= row.D_plot + 1e-9:
                tick_rows.append({
                    "match_id": row.match_id,
                    "x": 100.0 * t_plot / row.D_plot,
                    "label": str(m),     # keep momentum labels numeric here
                })
    ticks_min = pd.DataFrame(tick_rows)

    # special ticks at period ends in plot time, labeled 'period_end'
    # include p1 end if period 2 exists; p2 end if period 3 exists; p3 end if period 4 exists
    special_rows = []
    for row in meta_ticks.itertuples(index=False):
        if row.D_plot <= 0:
            continue
        if row.latest_period >= 2:  # halftime
            special_rows.append({
                "match_id": row.match_id,
                "x": 100.0 * (row.off_p2_plot / row.D_plot),
                "label": "period_end",
            })
        if row.latest_period >= 3:  # full time (only if extra time underway)
            special_rows.append({
                "match_id": row.match_id,
                "x": 100.0 * (row.off_p3_plot / row.D_plot),
                "label": "period_end",
            })
        if row.latest_period >= 4:  # extra time halftime
            special_rows.append({
                "match_id": row.match_id,
                "x": 100.0 * (row.off_p4_plot / row.D_plot),
                "label": "period_end",
            })
    ticks_special = pd.DataFrame(special_rows)

    ticks = (
        pd.concat([ticks_min, ticks_special], ignore_index=True)
          .drop_duplicates(["match_id", "x", "label"])
          .sort_values(["match_id", "x"], kind="stable")
          .reset_index(drop=True)
    )
    ticks["type"] = "tick"
    for c in ["team_id", "home", "period_id", "time_bin", "x0", "x1", "y0", "y1", "bar_color", "bar_fill", "pctl", "x"]:
        if c not in ticks.columns:
            ticks[c] = np.nan

    # assemble tidy output
    frames = [base_out[out_cols]]
    if not top.empty:
        frames.append(top[out_cols])
    if not ticks.empty:
        frames.append(ticks[out_cols])

    out = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["match_id", "type", "x0", "x"], kind="stable")
          .reset_index(drop=True)
    )

    # narrow dtypes
    for c in ("x0", "x1", "y0", "y1", "pctl", "x"):
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    for c in ("period_id", "time_bin"):
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int16")

    return out


# =====================================================================
# xG step
# =====================================================================

def df_match_xg_step(
    df: pd.DataFrame,
    *,
    meta: Optional[pd.DataFrame] = None,         # optional precomputed meta (no plot gaps necessary)
    per_bounds: Optional[pd.DataFrame] = None,   # optional precomputed period bounds
) -> pd.DataFrame:
    """
    build a tidy dataframe for the xg step chart

    types produced:
      'shot'  → the step series (includes start at (0,0) and end sentinel)
      'goal'  → goal markers with soccer-style minute labels
      'tick'  → vertical minute markers (including special 'period_end' markers)
      'ytick' → horizontal guides at 0.0, 1.0, 2.0, ... on the same 0–100 y scale

    columns (some are NaN by type):
      type, match_id, team_id, home, x, y, label, player_name, minute_label, line_color, point_color
    """
    # minimal columns and output shape
    need = {"match_id", "period_id", "time_seconds"}
    cols_out = [
        "type", "match_id", "team_id", "home", "x", "y", "label",
        "player_name", "minute_label", "line_color", "point_color"
    ]
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=cols_out)

    # timing meta (reuse if provided)
    if meta is None or per_bounds is None:
        meta, per_bounds = compute_timing_meta(df, bin_seconds=120, plot_gap_seconds=None)

    # shots only (keep a minimal set)
    keep = ["match_id", "team_id", "home", "period_id", "time_seconds", "xg", "goal"]
    if "player_name" in df.columns:
        keep.append("player_name")

    if "type_name" in df.columns:
        shots_src = df.loc[df["type_name"] == "shot", keep]
    else:
        shots_src = df.iloc[0:0][keep] if set(keep).issubset(df.columns) else pd.DataFrame(columns=keep)

    shots = (
        shots_src.dropna(subset=["period_id", "time_seconds", "xg"])
                 .copy()
                 .sort_values(["match_id", "team_id", "period_id", "time_seconds"], kind="stable")
    )
    if shots.empty:
        return pd.DataFrame(columns=cols_out)

    # clean numeric columns
    shots["xg"]        = pd.to_numeric(shots["xg"], errors="coerce").fillna(0.0)
    shots["goal"]      = pd.to_numeric(shots["goal"], errors="coerce").fillna(0).astype(int)
    shots["period_id"] = pd.to_numeric(shots["period_id"], errors="coerce").astype(int)

    # within-period time using canonical starts
    base_start = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0}
    shots["period_start"] = shots["period_id"].map(base_start).fillna(0.0).astype(float)
    shots["t_rel"] = shots["time_seconds"] - shots["period_start"]

    # cumulatives by (game, team)
    shots["xg_cum"]    = shots.groupby(["match_id", "team_id"], observed=True, sort=False)["xg"].cumsum()
    shots["goals_cum"] = shots.groupby(["match_id", "team_id"], observed=True, sort=False)["goal"].cumsum()

    # absolute time and normalized axes
    shots = shots.merge(meta[["match_id", "D", "off_p2", "off_p3", "off_p4"]], on="match_id", how="left")

    pid = shots["period_id"].to_numpy()
    off = np.where(pid == 2, shots["off_p2"].to_numpy(),
          np.where(pid == 3, shots["off_p3"].to_numpy(),
          np.where(pid == 4, shots["off_p4"].to_numpy(), 0.0)))
    shots["t_abs"] = shots["t_rel"].to_numpy() + off

    shots["x"] = (100.0 * shots["t_abs"] / shots["D"]).clip(0.0, 100.0)

    peak = shots.groupby("match_id", observed=True, sort=False)["xg_cum"].transform("max")
    shots["y_max"] = np.where(peak > 2.0, peak + 0.1, 2.0)
    shots["y"]     = (100.0 * shots["xg_cum"] / shots["y_max"]).clip(0.0, 100.0)

    # step series with start and observed end sentinel
    teams = (
        df[["match_id", "team_id", "home"]]
          .dropna(subset=["match_id", "team_id", "home"])
          .drop_duplicates()
    )

    meta_copy = meta.copy()
    meta_copy["T_obs_abs"] = meta_copy[["L1_obs", "L2_obs", "L3_obs", "L4_obs"]].sum(axis=1)
    meta_copy["x_obs_end"] = np.where(meta_copy["D"] > 0.0, 100.0 * meta_copy["T_obs_abs"] / meta_copy["D"], 0.0)

    core = shots[["match_id", "team_id", "home", "x", "y"]].copy()
    start_rows = teams.assign(x=0.0, y=0.0)

    last_rows = (
        shots.sort_values(["match_id", "team_id", "t_abs"], kind="stable")
             .groupby(["match_id", "team_id"], as_index=False, observed=True, sort=False)
             .tail(1)[["match_id", "team_id", "y"]]
    )
    end_rows = (
        teams.merge(last_rows, on=["match_id", "team_id"], how="left")
             .merge(meta_copy[["match_id", "x_obs_end"]], on="match_id", how="left")
             .assign(y=lambda d: d["y"].fillna(0.0), x=lambda d: d["x_obs_end"])
             [["match_id", "team_id", "home", "x", "y"]]
    )

    steps = (
        pd.concat([core, start_rows, end_rows], ignore_index=True)
          .sort_values(["match_id", "team_id", "x"], kind="stable")
          .reset_index(drop=True)
    )
    steps["line_color"] = np.where(steps["home"].eq("home"), futicolor.blue, futicolor.pink)

    # goals with soccer-style minute label
    if "player_name" not in shots.columns:
        shots["player_name"] = np.nan

    gsrc = shots.loc[
        shots["goal"] > 0,
        ["match_id", "team_id", "home", "period_id", "t_rel", "x", "y", "player_name"]
    ]

    base_min = {1: 0, 2: 45, 3: 90, 4: 105}
    nom_min  = {1: 45, 2: 45, 3: 15, 4: 15}
    def _minute_label(period_id: int, t_rel_sec: float) -> str:
        base = base_min.get(period_id, 0); nom = nom_min.get(period_id, 45)
        if t_rel_sec < nom * 60.0:
            return f"{base + int(t_rel_sec // 60) + 1}'"
        return f"{base + nom}+{int((t_rel_sec - nom * 60.0) // 60) + 1}'"

    goals = gsrc.copy()
    goals["minute_label"] = [
        _minute_label(int(p), float(t)) for p, t in zip(goals["period_id"], goals["t_rel"])
    ]
    goals = goals[["match_id", "team_id", "home", "x", "y", "player_name", "minute_label"]].reset_index(drop=True)
    goals["point_color"] = np.where(goals["home"].eq("home"), futicolor.blue, futicolor.pink)

    # vertical minute ticks (absolute time), with apostrophes and a special 'period_end' family
    # include 0' so the axis labeling can show the kickoff
    minutes = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135)
    per_game = meta_copy[["match_id", "D", "len_p1", "len_p2", "len_p3", "len_p4", "latest_period"]].copy()

    def _minute_to_abs(minute: int, len1: float, len2: float, len3: float) -> float | None:
        if minute <= 45:  return minute * 60.0
        if minute <= 90:  return len1 + (minute - 45) * 60.0
        if minute <= 105: return len1 + len2 + (minute - 90) * 60.0
        if minute <= 120: return len1 + len2 + len3 + (minute - 105) * 60.0
        return None

    tick_rows = []
    for row in per_game.itertuples(index=False):
        if row.D <= 0:
            continue
        for m in minutes:
            t_abs = _minute_to_abs(m, row.len_p1, row.len_p2, row.len_p3)
            if t_abs is not None and t_abs <= row.D + 1e-9:
                tick_rows.append({"match_id": row.match_id, "x": 100.0 * t_abs / row.D, "label": f"{m}'"})
    ticks_min = pd.DataFrame(tick_rows)

    # special ticks at period ends in absolute time, labeled 'period_end'
    # include p1 end if period 2 exists; p2 end if period 3 exists; p3 end if period 4 exists

    # observed ends from per_bounds (t1 in absolute seconds)
    p1_end = per_bounds.loc[per_bounds["period_id"] == 1, ["match_id", "t1"]].rename(columns={"t1": "p1_end"})
    p2_end = per_bounds.loc[per_bounds["period_id"] == 2, ["match_id", "t1"]].rename(columns={"t1": "p2_end"})
    p3_end = per_bounds.loc[per_bounds["period_id"] == 3, ["match_id", "t1"]].rename(columns={"t1": "p3_end"})

    ends = (
        meta_copy[["match_id", "D", "latest_period", "len_p1", "len_p2", "len_p3"]]
        .merge(p1_end, on="match_id", how="left")
        .merge(p2_end, on="match_id", how="left")
        .merge(p3_end, on="match_id", how="left")
    )

    special_rows = []
    for row in ends.itertuples(index=False):
        if row.D <= 0:
            continue
        # halftime → end of p1 if p2 exists
        if row.latest_period >= 2:
            t = float(row.p1_end) if pd.notna(row.p1_end) else float(row.len_p1)
            special_rows.append({"match_id": row.match_id, "x": 100.0 * (t / row.D), "label": "period_end"})
        # full time → end of p2 only if extra time exists
        if row.latest_period >= 3:
            fallback = float(row.len_p1) + float(row.len_p2)
            t = float(row.p2_end) if pd.notna(row.p2_end) else fallback
            special_rows.append({"match_id": row.match_id, "x": 100.0 * (t / row.D), "label": "period_end"})
        # extra time halftime → end of p3 only if p4 exists
        if row.latest_period >= 4:
            fallback = float(row.len_p1) + float(row.len_p2) + float(row.len_p3)
            t = float(row.p3_end) if pd.notna(row.p3_end) else fallback
            special_rows.append({"match_id": row.match_id, "x": 100.0 * (t / row.D), "label": "period_end"})

    ticks_special = pd.DataFrame(special_rows)

    ticks = (
        pd.concat([ticks_min, ticks_special], ignore_index=True)
          .drop_duplicates(["match_id", "x", "label"])
          .sort_values(["match_id", "x"], kind="stable")
          .reset_index(drop=True)
    )

    # horizontal y-axis guides, including baseline 0.0, with labels carried in the data
    ycap = shots.groupby("match_id", observed=True, sort=False)["y_max"].max().reset_index(name="y_cap")
    ytick_rows = []

    for row in ycap.itertuples(index=False):
        # baseline 0.0
        ytick_rows.append({"match_id": row.match_id, "y": 0.0, "label": "0.0"})

        cap = float(row.y_cap)
        if np.isfinite(cap) and cap > 0.0:
            kmax = int(np.floor(cap))  # whole xg guides above zero
            for k in range(1, kmax + 1):
                yv = 100.0 * (k / cap)
                ytick_rows.append({"match_id": row.match_id, "y": yv, "label": f"{k:.1f}"})

    yticks = pd.DataFrame(ytick_rows, columns=["match_id", "y", "label"])

    # assemble tidy output (align columns)
    steps_out = steps.assign(
        type="shot", label=np.nan, player_name=np.nan, minute_label=np.nan, point_color=np.nan
    )
    ticks_out = ticks.assign(
        type="tick", team_id=np.nan, home=np.nan, y=np.nan,
        player_name=np.nan, minute_label=np.nan, line_color=np.nan, point_color=np.nan
    )
    goals_out = goals.assign(type="goal", label=np.nan, line_color=np.nan)
    yticks_out = yticks.assign(
        type="ytick", team_id=np.nan, home=np.nan, x=np.nan,
        player_name=np.nan, minute_label=np.nan, line_color=np.nan, point_color=np.nan
    )

    for d in (steps_out, ticks_out, goals_out, yticks_out):
        for c in cols_out:
            if c not in d.columns:
                d[c] = np.nan

    out = (
        pd.concat([steps_out[cols_out], ticks_out[cols_out], goals_out[cols_out], yticks_out[cols_out]], ignore_index=True)
          .sort_values(["match_id", "x", "type"], kind="stable")
          .reset_index(drop=True)
    )

    # narrow dtypes
    for c in ("x", "y"):
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    return out


# =====================================================================
# territory
# =====================================================================

def df_match_territory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build territory bins for plotting.

    Returns one row per (game, zone) with:
      match_id, x_min, x_max, y_min, y_max, team_label, possession_label, fill_color.
    """
    need = {"match_id", "team_id", "home", "type_name", "x_start", "y_start", "phase_team"}
    out_cols = [
        "match_id", "x_min", "x_max", "y_min", "y_max",
        "team_label", "possession_label", "fill_color",
    ]
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=out_cols)

    # in-possession, open-play (drop restarts) and common home frame
    exclude = {"corner_crossed", "corner_short", "throw_in", "goal_kick"}
    df_poss = df.loc[
        (df["team_id"] == df["phase_team"]) & (~df["type_name"].isin(exclude))
    ].copy()
    if df_poss.empty:
        return pd.DataFrame(columns=out_cols)

    pitch_specs = {"pitch_length": 105.0, "pitch_width": 68.0, "orientation": "wide"}
    home = transform_xy(df_poss[df_poss["home"] == "home"], pitch=pitch_specs, flip_coords=False)
    away = transform_xy(df_poss[df_poss["home"] == "away"], pitch=pitch_specs, flip_coords=True)

    xcol, ycol = "x_start", "y_start"
    home_v = home[["match_id", xcol, ycol]].dropna(subset=[xcol, ycol]).assign(side="home")
    away_v = away[["match_id", xcol, ycol]].dropna(subset=[xcol, ycol]).assign(side="away")
    both = pd.concat([home_v, away_v], ignore_index=True)
    if both.empty:
        return pd.DataFrame(columns=out_cols)

    # zone edges and fast binning
    length_zones = np.array([0.0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0], dtype=float)
    width_zones  = np.array([0.0, 13.84, 24.84, 43.16, 54.16, 68.0], dtype=float)
    nx, ny = len(length_zones) - 1, len(width_zones) - 1

    xb = np.searchsorted(length_zones, both[xcol].to_numpy(copy=False), side="right") - 1
    yb = np.searchsorted(width_zones,  both[ycol].to_numpy(copy=False),  side="right") - 1
    np.clip(xb, 0, nx - 1, out=xb)
    np.clip(yb, 0, ny - 1, out=yb)
    both["x_bin"] = xb.astype(np.int16)
    both["y_bin"] = yb.astype(np.int16)

    cnt = (
        both.groupby(["match_id", "x_bin", "y_bin", "side"], observed=True, sort=False)
            .size()
            .unstack("side", fill_value=0)
            .rename(columns={"home": "touches_home", "away": "touches_away"})
            .reset_index()
    )
    tot = cnt["touches_home"] + cnt["touches_away"]
    cnt["poss_home_pct"] = np.where(tot > 0, cnt["touches_home"] / tot, 0.5)

    # rectangle coordinates
    xb = cnt["x_bin"].to_numpy(copy=False)
    yb = cnt["y_bin"].to_numpy(copy=False)
    cnt["x_min"] = length_zones[xb]
    cnt["x_max"] = length_zones[xb + 1]
    cnt["y_min"] = width_zones[yb]
    cnt["y_max"] = width_zones[yb + 1]

    # optional team names
    if "team_name" in df.columns:
        names = (
            df[["match_id", "home", "team_name"]]
              .dropna(subset=["home", "team_name"])
              .drop_duplicates(["match_id", "home"])
              .pivot(index="match_id", columns="home", values="team_name")
              .rename(columns={"home": "home_team_name", "away": "away_team_name"})
              .reset_index()
        )
        cnt = cnt.merge(names, on="match_id", how="left")

    if "home_team_name" not in cnt.columns:
        cnt["home_team_name"] = "home"
    if "away_team_name" not in cnt.columns:
        cnt["away_team_name"] = "away"

    home_dom = cnt["poss_home_pct"] >= 0.50
    poss_for_label = np.where(home_dom, cnt["poss_home_pct"], 1.0 - cnt["poss_home_pct"])

    cnt["team_label"] = np.where(home_dom, cnt["home_team_name"], cnt["away_team_name"])
    cnt["possession_label"] = (np.round(poss_for_label * 100).astype(int)).astype(str) + "%"

    cnt["fill_color"] = np.where(
        home_dom,
        np.where(poss_for_label <= 0.55, futicolor.blue1, futicolor.blue),
        np.where(poss_for_label <= 0.55, futicolor.pink1, futicolor.pink),
    )

    out = (
        cnt[["match_id", "x_min", "x_max", "y_min", "y_max",
             "team_label", "possession_label", "fill_color"]]
        .sort_values(["match_id", "y_min", "x_min"], kind="stable")
        .reset_index(drop=True)
    )
    # narrow numeric rectangles to float32
    for c in ("x_min", "x_max", "y_min", "y_max"):
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


# =====================================================================
# shotmap
# =====================================================================

def df_match_shotmap(
    df: pd.DataFrame,
    *,
    shot_percentiles_path: str | None = None,
) -> pd.DataFrame:
    """
    Build a compact shots dataframe ready for plotting

    Coordinate systems returned:
      • (x_both, y_both): wide-pitch frame
            Home: (x, y) -> (x, W - y)
            Away: (x, y) -> (L - x, y)
      • (x_team, y_team): team-attacking-top frame for half-pitch plots
            Home: X = 68 - y_both,       Y = x_both
            Away: X = y_both,            Y = 105 - x_both
      • end-point columns (xend_both, yend_both, xend_team, yend_team) are populated only for goals

    Also returns:
      - shot_size in [0,1] from xG percentiles
      - point_color (team fill), edge_color (white for goals; blue/pink otherwise)
      - alpha (0.0 off-target; 0.4 on-target no-goal; 0.6 for goals)
    """
    # ---- base columns ----
    # include end coordinates if present in source data
    keep = [
        "match_id", "team_id", "home",
        "x_start", "y_start", "x_end", "y_end",
        "xg", "goal", "result_name",
    ]
    cols_present = [c for c in keep if c in df.columns]

    # select shots with valid xg and starting coordinates
    shots = (
        df.loc[df.get("xg") > 0, cols_present]
          .dropna(subset=["x_start", "y_start", "xg"], how="any")
          .copy()
          .reset_index(drop=True)
    )

    # early return with correct schema if empty
    if shots.empty:
        return pd.DataFrame(columns=[
            "match_id","team_id","home",
            "x_both","y_both","x_team","y_team",
            "xend_both","yend_both","xend_team","yend_team",
            "xg","goal","on_target",
            "shot_size","point_color","edge_color","alpha",
        ])

    # ---- dtypes ----
    # coerce numeric types for coordinate and xg fields
    for c in ("x_start", "y_start", "xg"):
        shots[c] = pd.to_numeric(shots[c], errors="coerce").astype("float32")

    # ensure end columns exist so downstream logic is uniform
    for c in ("x_end", "y_end"):
        if c not in shots.columns:
            shots[c] = np.nan
        shots[c] = pd.to_numeric(shots[c], errors="coerce").astype("float32")

    # normalize goal to int8
    if "goal" not in shots.columns:
        shots["goal"] = 0
    shots["goal"] = pd.to_numeric(shots["goal"], errors="coerce").fillna(0).astype("int8")

    # ---- constants and masks ----
    L, W = 105.0, 68.0  # pitch length and width in meters
    is_home = shots["home"].astype(str).str.lower().eq("home").to_numpy()
    is_away = ~is_home
    is_goal = shots["goal"].to_numpy(dtype=bool)

    # raw coordinate arrays for starts and ends
    x_raw = shots["x_start"].to_numpy(dtype=float, copy=True)
    y_raw = shots["y_start"].to_numpy(dtype=float, copy=True)
    x_end_raw = shots["x_end"].to_numpy(dtype=float, copy=True)
    y_end_raw = shots["y_end"].to_numpy(dtype=float, copy=True)

    # ---- wide-frame coords for starting point (x_both, y_both) ----
    # away mirrors x across midfield, home inverts y across halfway line
    x_both = x_raw.copy()
    y_both = y_raw.copy()
    x_both[is_away] = L - x_both[is_away]
    y_both[is_home] = W - y_both[is_home]

    shots["x_both"] = x_both.astype("float32")
    shots["y_both"] = y_both.astype("float32")

    # keep legacy names pointing to wide-frame start coordinates for compatibility
    shots["x_start"] = shots["x_both"]
    shots["y_start"] = shots["y_both"]

    # ---- wide-frame coords for end point (xend_both, yend_both) ----
    # apply the same mapping as for starts, then null out non-goals
    xend_both = x_end_raw.copy()
    yend_both = y_end_raw.copy()
    xend_both[is_away] = L - xend_both[is_away]
    yend_both[is_home] = W - yend_both[is_home]

    # only keep end coordinates for goals
    xend_both[~is_goal] = np.nan
    yend_both[~is_goal] = np.nan

    shots["xend_both"] = xend_both.astype("float32")
    shots["yend_both"] = yend_both.astype("float32")

    # ---- team-attacking-top coords for start (x_team, y_team) ----
    # home:  X = 68 - y_both,  Y = x_both
    # away:  X = y_both,       Y = 105 - x_both
    x_team = np.empty_like(x_both, dtype=float)
    y_team = np.empty_like(y_both, dtype=float)
    x_team[is_home] = W - y_both[is_home]
    y_team[is_home] = x_both[is_home]
    x_team[is_away] = y_both[is_away]
    y_team[is_away] = L - x_both[is_away]

    shots["x_team"] = x_team.astype("float32")
    shots["y_team"] = y_team.astype("float32")

    # ---- team-attacking-top coords for end (xend_team, yend_team) ----
    # transform from the wide-frame end coordinates to the team frame using the same rules
    xend_team = np.empty_like(xend_both, dtype=float)
    yend_team = np.empty_like(yend_both, dtype=float)
    xend_team[is_home] = W - yend_both[is_home]
    yend_team[is_home] = xend_both[is_home]
    xend_team[is_away] = yend_both[is_away]
    yend_team[is_away] = L - xend_both[is_away]

    shots["xend_team"] = xend_team.astype("float32")
    shots["yend_team"] = yend_team.astype("float32")

    # ---- percentile-based marker size (0..1) from calibration csv ----
    qvec = _load_shots_xg_percentiles_cached(shot_percentiles_path)
    xg = shots["xg"].to_numpy(dtype=float)
    if qvec is not None and np.isfinite(qvec).all() and qvec.max() > qvec.min():
        qvec = np.maximum.accumulate(qvec)  # enforce monotone
        pgrid = np.linspace(0.0, 1.0, 101)
        shot_size = np.clip(np.interp(xg, qvec, pgrid, left=0.0, right=1.0), 0.0, 1.0)
    else:
        sx = np.sort(xg[np.isfinite(xg)])
        n = float(len(sx)) if sx.size else 1.0
        shot_size = (np.searchsorted(sx, xg, side="right") / n).clip(0.0, 1.0) if sx.size else np.zeros_like(xg)
    shots["shot_size"] = shot_size.astype("float32")

    # ---- on-target flag from result_name or goal ----
    rn = shots["result_name"].astype(str).str.strip().str.lower() if "result_name" in shots.columns else ""
    on_from_result = rn.eq("ontarget").to_numpy() if isinstance(rn, pd.Series) else np.zeros(len(shots), dtype=bool)
    on_target = is_goal | on_from_result
    shots["on_target"] = on_target

    # ---- colors and alpha ----
    shots["point_color"] = np.where(is_home, futicolor.blue, futicolor.pink)
    shots["edge_color"]  = np.where(is_goal, futicolor.light,
                            np.where(is_home, futicolor.blue, futicolor.pink))
    # alpha scales with event salience
    alpha = np.where(is_goal, 0.6, np.where(on_target, 0.4, 0.0)).astype("float32")
    shots["alpha"] = alpha

    # ---- sort so larger markers draw underneath smaller ones for better legibility ----
    shots = shots.sort_values(["match_id", "shot_size"], ascending=[True, False], kind="stable").reset_index(drop=True)

    # ---- final column order ----
    cols_out = [
        "match_id","team_id","home",
        "x_both","y_both","x_team","y_team",
        "xend_both","yend_both","xend_team","yend_team",
        "xg","goal","on_target",
        "shot_size","point_color","edge_color","alpha",
    ]
    for c in cols_out:
        if c not in shots.columns:
            shots[c] = np.nan
    shots["alpha"] = pd.to_numeric(shots["alpha"], errors="coerce").astype("float32")
    return shots[cols_out]


# =====================================================================
# action heatmap (percentile-scaled α in [0,1])
# =====================================================================

def df_match_action_heatmap(
    df: pd.DataFrame,
    *,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    length_edges: int = 21,   # edges count (→ nx = 20 cells by default)
    width_edges: int = 14,    # edges count (→ ny = 13 cells by default)
    sigma: float | None = None,
    both_scaler_path: str | None = None,
    team_scaler_path: str | None = None,
    alpha_gamma: float = 1.8, # gamma for alpha shaping (>= 1.0 darkens lows)
) -> pd.DataFrame:
    """
    Build a match-level action heatmap with two independent blur/aggregate paths:

      1) Combined path ("both"): sum home+away raw counts per bin, then blur once.
      2) Per-team path ("home","away"): blur home-only and away-only counts independently.

    Then convert each blurred grid to per-minute and map values to α ∈ [0,1] using
    percentile-based scalers:
      * 'both' scaler trained on combined per-minute blurred bins
      * 'team' scaler trained on single-team per-minute blurred bins

    Returns one row per spatial bin with columns:
      x_min, x_max, y_min, y_max,
      both_count, home_count, away_count,
      both_fill, home_fill, away_fill

    Notes:
      - Coordinates are mapped into the "match frame":
          home: (x,y) unchanged
          away: (x,y) -> (L - x, W - y)
      - length_edges and width_edges are numbers of edges, not cells.
      - sigma is in "cells" (same units used when saving the scalers). If None,
        we use max(1, min(nx,ny)/10), which must match the training rule.
      - `alpha_gamma`: post-processing power transform on α to adjust contrast.
        1.0 → no change; 1.2–1.8 recommended to darken low-activity bins.
    """
    # ----- required columns and early exit -----
    need = {"home", "x_start", "y_start", "time_seconds"}
    empty_cols = [
        "x_min","x_max","y_min","y_max",
        "both_count","home_count","away_count",
        "both_fill","home_fill","away_fill",
    ]
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=empty_cols)

    src = (
        df.loc[:, ["home", "x_start", "y_start", "time_seconds"]]
          .dropna(subset=["x_start","y_start","time_seconds"])
          .copy()
    )
    if src.empty:
        return pd.DataFrame(columns=empty_cols)

    # ----- grid, geometry, and minutes played -----
    L, W = float(pitch_length), float(pitch_width)
    lx, wy = int(length_edges), int(width_edges)
    if lx < 2 or wy < 2:
        raise ValueError("length_edges and width_edges must be >= 2 (edges, not cells).")

    length_z = np.linspace(0.0, L, lx, dtype=float)
    width_z  = np.linspace(0.0, W, wy, dtype=float)
    nx, ny = lx - 1, wy - 1

    # rectangle geometry matching histogram order after transpose
    X_min, Y_min = np.meshgrid(length_z[:-1], width_z[:-1], indexing="xy")
    X_max, Y_max = np.meshgrid(length_z[1:],  width_z[1:],  indexing="xy")
    x_min = X_min.ravel().astype(float)
    x_max = X_max.ravel().astype(float)
    y_min = Y_min.ravel().astype(float)
    y_max = Y_max.ravel().astype(float)

    # observed match minutes
    tmax = pd.to_numeric(src["time_seconds"], errors="coerce").max()
    if not np.isfinite(tmax) or tmax <= 0:
        base = pd.DataFrame({
            "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "both_count": 0.0, "home_count": 0.0, "away_count": 0.0,
            "both_fill": 0.0, "home_fill": 0.0, "away_fill": 0.0,
        })
        for c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").astype("float32")
        return base

    minutes = float(tmax) / 60.0

    # ----- map into match frame -----
    is_away = src["home"].astype(str).str.strip().str.lower().eq("away").to_numpy()
    x_raw = pd.to_numeric(src["x_start"], errors="coerce").to_numpy(dtype=float, copy=True)
    y_raw = pd.to_numeric(src["y_start"], errors="coerce").to_numpy(dtype=float, copy=True)

    x_map = x_raw.copy()
    y_map = y_raw.copy()
    x_map[is_away] = L - x_map[is_away]
    y_map[is_away] = W - y_map[is_away]

    # ----- histogram counts per team -----
    xh = x_map[~is_away]; yh = y_map[~is_away]
    H_home, _, _ = np.histogram2d(xh, yh, bins=[length_z, width_z])   # (nx, ny)
    C_home = H_home.T                                                 # (ny, nx)

    xa = x_map[is_away]; ya = y_map[is_away]
    H_away, _, _ = np.histogram2d(xa, ya, bins=[length_z, width_z])   # (nx, ny)
    C_away = H_away.T                                                 # (ny, nx)

    C_both = C_home + C_away

    # ----- Gaussian blur in the two paths -----
    sigma_used = float(sigma) if sigma is not None else max(1.0, min(nx, ny) / 10.0)
    if sigma_used > 0.0:
        I_home = gaussian_filter(C_home, sigma=sigma_used)
        I_away = gaussian_filter(C_away, sigma=sigma_used)
        I_both = gaussian_filter(C_both, sigma=sigma_used)
    else:
        I_home, I_away, I_both = C_home, C_away, C_both

    # ----- per-minute -----
    I_home_pm = (I_home / minutes).astype(float, copy=False)
    I_away_pm = (I_away / minutes).astype(float, copy=False)
    I_both_pm = (I_both / minutes).astype(float, copy=False)

    # ----- load percentile scalers and map to α ∈ [0,1] -----
    sc_both = _load_heatmap_scaler_both_cached(
        length_edges=lx, width_edges=wy, L=L, W=W, sigma=sigma_used, explicit_path=both_scaler_path
    )
    sc_team = _load_heatmap_scaler_team_cached(
        length_edges=lx, width_edges=wy, L=L, W=W, sigma=sigma_used, explicit_path=team_scaler_path
    )

    if sc_both:
        both_fill = apply_bin_scaler(I_both_pm.ravel(), sc_both).reshape(I_both_pm.shape)
    else:
        both_fill = np.zeros_like(I_both_pm)

    if sc_team:
        home_fill = apply_bin_scaler(I_home_pm.ravel(), sc_team).reshape(I_home_pm.shape)
        away_fill = apply_bin_scaler(I_away_pm.ravel(), sc_team).reshape(I_away_pm.shape)
    else:
        home_fill = np.zeros_like(I_home_pm)
        away_fill = np.zeros_like(I_away_pm)

    # --- optional gamma shaping of alpha to increase/decrease contrast ---
    # α' = α^gamma (gamma > 1 darkens low alphas, < 1 brightens)
    g = float(alpha_gamma)
    for arr in (both_fill, home_fill, away_fill):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(arr, 0.0, 1.0, out=arr)
        if np.isfinite(g) and abs(g - 1.0) > 1e-12:
            np.power(arr, max(g, 1e-6), out=arr)


    out = pd.DataFrame({
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
        "both_count": C_both.ravel().astype(float),
        "home_count": C_home.ravel().astype(float),
        "away_count": C_away.ravel().astype(float),
        "both_fill": both_fill.ravel().astype(float),
        "home_fill": home_fill.ravel().astype(float),
        "away_fill": away_fill.ravel().astype(float),
    })

    out = out.sort_values(["y_min", "x_min"], kind="stable").reset_index(drop=True)

    for c in ("x_min","x_max","y_min","y_max",
              "both_count","home_count","away_count",
              "both_fill","home_fill","away_fill"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    return out

# %%
