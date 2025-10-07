#!/usr/bin/env python3
"""
build and save calibration csvs for:
  • momentum (per-bin abs(max prevalue))
  • shot xg (for point sizes)
  • heatmap scalers (percentile-based, per-minute blurred values) for:
      - combined teams ("both" scaler)
      - single-team ("team" scaler)

outputs (by default):
  futiplot/data/momentum_prevalue_percentiles_<BIN>.csv
  futiplot/data/shots_xg_percentiles.csv
  futiplot/data/heatmap_scaler_both_L<L>_W<W>_Z<lx>x<ly>_sigma<S>.csv
  futiplot/data/heatmap_scaler_team_L<L>_W<W>_Z<lx>x<ly>_sigma<S>.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from futiplot.match_card.calibrate_percentiles import (
    compute_and_save_momentum_percentiles_csv,
    compute_and_save_shot_xg_percentiles_csv,
    compute_and_save_bin_scaler_csv,
)

# minimal columns needed across all calibrations, including heatmap scalers
NEEDED_COLS = {
    "match_id", "team_id", "home",
    "period_id", "time_seconds",
    "type_name", "prev_scores", "prev_concedes",
    "x_start", "y_start", "xg",
}


def load_events(path: str) -> pd.DataFrame:
    """load events from csv/parquet, requesting only needed columns when possible"""
    ext = os.path.splitext(path)[1].lower()
    usecols_list = list(NEEDED_COLS)
    try:
        if ext in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path, columns=usecols_list)
            except Exception:
                return pd.read_parquet(path)
        else:
            want = set(usecols_list)
            return pd.read_csv(path, usecols=lambda c: (c in want))
    except Exception as e:
        print(f"error: failed to load events from {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def _grid_edges(L: float, W: float, n_len_edges: int, n_wid_edges: int) -> Tuple[np.ndarray, np.ndarray]:
    """build uniform bin edges along length and width"""
    length_edges = np.linspace(0.0, float(L), int(n_len_edges))
    width_edges = np.linspace(0.0, float(W), int(n_wid_edges))
    return length_edges, width_edges


def _counts_for_team(
    block: pd.DataFrame,
    L: float,
    W: float,
    length_edges: np.ndarray,
    width_edges: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    counts for one team in match frame with your flip rules
      - home: keep raw x and raw y
      - away: x -> L - x, y -> W - y
    returns a (ny, nx) array (rows=y bins, cols=x bins)
    """
    if block.empty:
        return np.zeros((ny, nx), dtype=float)

    x = pd.to_numeric(block["x_start"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(block["y_start"], errors="coerce").to_numpy(dtype=float)
    is_away = str(block["home"].iloc[0]).strip().lower() == "away"

    if is_away:
        x = L - x
        y = W - y

    H, _, _ = np.histogram2d(x, y, bins=[length_edges, width_edges])  # shape (nx, ny)
    return H.T  # shape (ny, nx)


def _collect_heatmap_training_vectors(
    df: pd.DataFrame,
    *,
    L: float,
    W: float,
    n_len_edges: int,
    n_wid_edges: int,
    sigma: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    build training vectors for the two heatmap scalers

    returns
      v_both: concatenated per-minute blurred values for combined teams
      v_team: concatenated per-minute blurred values for single teams (home and away)
    """
    # keep only rows with essential fields and finite coordinates
    need = {"match_id", "team_id", "home", "time_seconds", "x_start", "y_start"}
    if df.empty or not need.issubset(df.columns):
        return np.array([], dtype=float), np.array([], dtype=float)

    x = df.loc[:, list(need)].dropna(subset=["x_start", "y_start", "time_seconds"]).copy()
    if x.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    # build grid and derived sizes
    length_edges, width_edges = _grid_edges(L, W, n_len_edges, n_wid_edges)
    nx = len(length_edges) - 1
    ny = len(width_edges) - 1

    # default sigma if not provided
    if sigma is None:
        sigma_used = max(1.0, min(nx, ny) / 10.0)
    else:
        sigma_used = float(sigma)

    v_both_list: list[np.ndarray] = []
    v_team_list: list[np.ndarray] = []

    # iterate by match to compute per-match grids and per-minute values
    for mid, m in x.groupby("match_id", observed=True, sort=False):
        # time normalization in minutes played so far
        tmax = pd.to_numeric(m["time_seconds"], errors="coerce").max()
        if not np.isfinite(tmax) or tmax <= 0:
            continue
        minutes = float(tmax) / 60.0

        # split home/away
        m_home = m.loc[m["home"].astype(str).str.lower().eq("home")]
        m_away = m.loc[m["home"].astype(str).str.lower().eq("away")]

        # per-team count grids
        c_home = _counts_for_team(m_home, L, W, length_edges, width_edges, nx, ny)
        c_away = _counts_for_team(m_away, L, W, length_edges, width_edges, nx, ny)
        c_both = c_home + c_away

        # blur in the two paths
        i_home = gaussian_filter(c_home, sigma=sigma_used) if sigma_used > 0 else c_home
        i_away = gaussian_filter(c_away, sigma=sigma_used) if sigma_used > 0 else c_away
        i_both = gaussian_filter(c_both, sigma=sigma_used) if sigma_used > 0 else c_both

        # convert to per-minute
        i_home_pm = (i_home / minutes).ravel()
        i_away_pm = (i_away / minutes).ravel()
        i_both_pm = (i_both / minutes).ravel()

        # append to training vectors
        v_team_list.append(i_home_pm)
        v_team_list.append(i_away_pm)
        v_both_list.append(i_both_pm)

    v_both = np.concatenate(v_both_list) if v_both_list else np.array([], dtype=float)
    v_team = np.concatenate(v_team_list) if v_team_list else np.array([], dtype=float)
    # keep nonnegative finite values only
    v_both = v_both[np.isfinite(v_both) & (v_both >= 0.0)]
    v_team = v_team[np.isfinite(v_team) & (v_team >= 0.0)]
    return v_both, v_team


def main() -> None:
    ap = argparse.ArgumentParser(description="build momentum + shot xg percentiles and heatmap scalers")
    ap.add_argument("events_path", help="path to historical events (csv or parquet)")
    ap.add_argument("--bin-seconds", type=int, default=120, help="bin size for momentum (default: 120)")
    ap.add_argument(
        "--out-dir",
        default="futiplot/data",
        help="output directory for csvs (default: futiplot/data)",
    )
    # heatmap grid + pitch + blur
    ap.add_argument("--pitch-length", type=float, default=105.0, help="pitch length in meters (default: 105)")
    ap.add_argument("--pitch-width", type=float, default=68.0, help="pitch width in meters (default: 68)")
    ap.add_argument("--length-edges", type=int, default=21, help="number of length edges (default: 21 -> 20 cells)")
    ap.add_argument("--width-edges", type=int, default=14, help="number of width edges (default: 14 -> 13 cells)")
    ap.add_argument("--sigma", type=float, default=None, help="gaussian sigma in cells; default uses min(nx,ny)/10")

    args = ap.parse_args()

    df = load_events(args.events_path)
    if df.empty:
        print("error: no rows loaded or missing required columns.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # existing outputs
    mom_out = os.path.join(args.out_dir, f"momentum_prevalue_percentiles_{args.bin_seconds}.csv")
    xg_out = os.path.join(args.out_dir, "shots_xg_percentiles.csv")

    print(f"computing momentum prevalue percentiles (bin={args.bin_seconds}s)…")
    compute_and_save_momentum_percentiles_csv(df, mom_out, bin_seconds=args.bin_seconds)
    print(f"saved: {mom_out}")

    print("computing shot xg percentiles…")
    compute_and_save_shot_xg_percentiles_csv(df, xg_out)
    print(f"saved: {xg_out}")

    # new heatmap scalers
    print("collecting blurred per-minute values for heatmap scalers…")
    v_both, v_team = _collect_heatmap_training_vectors(
        df,
        L=float(args.pitch_length),
        W=float(args.pitch_width),
        n_len_edges=int(args.length_edges),
        n_wid_edges=int(args.width_edges),
        sigma=args.sigma,
    )

    if v_both.size == 0 or v_team.size == 0:
        print("warning: insufficient data to fit heatmap scalers (vectors are empty).", file=sys.stderr)
    else:
        # determine the sigma actually used for filenames
        nx = int(args.length_edges) - 1
        ny = int(args.width_edges) - 1
        sigma_used = float(args.sigma) if args.sigma is not None else max(1.0, min(nx, ny) / 10.0)

        both_out = os.path.join(
            args.out_dir,
            f"heatmap_scaler_both_L{int(args.pitch_length)}_W{int(args.pitch_width)}_"
            f"Z{int(args.length_edges)}x{int(args.width_edges)}_sigma{sigma_used:.3g}.csv",
        )
        team_out = os.path.join(
            args.out_dir,
            f"heatmap_scaler_team_L{int(args.pitch_length)}_W{int(args.pitch_width)}_"
            f"Z{int(args.length_edges)}x{int(args.width_edges)}_sigma{sigma_used:.3g}.csv",
        )

        print(f"fitting and saving heatmap scalers (both/team)…")
        compute_and_save_bin_scaler_csv(v_both, both_out)
        compute_and_save_bin_scaler_csv(v_team, team_out)
        print(f"saved: {both_out}")
        print(f"saved: {team_out}")

        # small summary for sanity check
        print(f"samples used — both: {v_both.size:,} values, team: {v_team.size:,} values")

    print("done.")


if __name__ == "__main__":
    main()
