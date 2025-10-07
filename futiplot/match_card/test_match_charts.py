#%%

#!/usr/bin/env python
# preview all match-card plots with a consistent dark theme

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Rectangle, FancyBboxPatch, PathPatch
from importlib import resources as ir

from futiplot.match_card.match_card_charts import (
    compute_timing_meta,
    df_match_momentum,
    df_match_xg_step,
    df_match_territory,
    df_match_action_heatmap,
    df_match_shotmap,
)
from futiplot.utils import futicolor, load_sample_data, transform_xy
from futiplot.soccer.pitch import plot_pitch

BIN = 120  # matches momentum_prevalue_percentiles_120.csv


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def pick_game(df: pd.DataFrame) -> int:
    return int(df["match_id"].dropna().iloc[0])

def load_shots_xg_percentiles() -> np.ndarray | None:
    """futiplot.data/shots_xg_percentiles.csv → array(101,) or None."""
    try:
        p = ir.files("futiplot.data") / "shots_xg_percentiles.csv"
        arr = np.loadtxt(p.open("rb"), delimiter=",", skiprows=1)
        vec = (arr[:, 1] if arr.ndim == 2 else np.array([arr[1]])).astype(float)
        return np.maximum.accumulate(vec) if vec.size == 101 else None
    except Exception:
        return None
    
def _rounded_end_bar(ax, *, x, w, y0, y1, color, z=2, radius=None, alpha=1.0):
    """
    Draw a vertical bar from y0 to y1 at x..x+w, with rounded corners only
    on the end away from y=50, and square corners at y=50.

    alpha controls fill transparency (0..1).
    """
    # guard
    if not np.isfinite([x, w, y0, y1]).all() or w <= 0 or abs(y1 - y0) <= 0:
        return

    y_low, y_high = (y0, y1) if y0 < y1 else (y1, y0)
    h = y_high - y_low

    # round the end farther from 50
    round_top = (y_high - 50) > (50 - y_low)

    # corner radius
    if radius is None:
        radius = min(0.45 * w, 0.6 * h)
    r = max(0.0, min(radius, h * 0.5, w * 0.5))
    k = 0.5522847498307936  # cubic-Bezier quarter circle

    verts, codes = [], []

    if round_top:
        verts.append((x, y_low));               codes.append(Path.MOVETO)
        verts.append((x, y_high - r));          codes.append(Path.LINETO)
        verts += [(x, y_high - r + k*r), (x + r - k*r, y_high), (x + r, y_high)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        verts.append((x + w - r, y_high));      codes.append(Path.LINETO)
        verts += [(x + w - r + k*r, y_high), (x + w, y_high - r + k*r), (x + w, y_high - r)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        verts.append((x + w, y_low));           codes.append(Path.LINETO)
        verts.append((x, y_low));               codes.append(Path.CLOSEPOLY)
    else:
        verts.append((x, y_high));              codes.append(Path.MOVETO)
        verts.append((x, y_low + r));           codes.append(Path.LINETO)
        verts += [(x, y_low + r - k*r), (x + r - k*r, y_low), (x + r, y_low)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        verts.append((x + w - r, y_low));       codes.append(Path.LINETO)
        verts += [(x + w - r + k*r, y_low), (x + w, y_low + r - k*r), (x + w, y_low + r)]
        codes += [Path.CURVE4, Path.CURVE4, Path.CURVE4]
        verts.append((x + w, y_high));          codes.append(Path.LINETO)
        verts.append((x, y_high));              codes.append(Path.CLOSEPOLY)

    patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=z, alpha=float(alpha))
    ax.add_patch(patch)


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------

def plot_momentum(df_game: pd.DataFrame, meta: pd.DataFrame) -> None:
    mm = df_match_momentum(
        df_game,
        bin_seconds=BIN,
        percentiles_path=None,
        smoothing=1,
        width_scale=0.85,
        meta=meta,
    )

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=futicolor.dark)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

    # baseline behind bars
    ax.axhline(50, lw=1, color="#BBBBBB", zorder=0.1)

    # ticks: draw only 'period_end' lines; put minute labels below axis (no lines)
    ticks = mm[mm["type"] == "tick"]

    # vertical guide for period ends
    for _, r in ticks.iterrows():
        if str(r.get("label")) == "period_end":
            y0, y1 = ax.get_ylim()
            ax.vlines(
                r["x"],
                ymin=-3, ymax=y1,
                lw=2,
                color=futicolor.light,
                clip_on=False,
                zorder=0
            )

    # minute labels under the axis
    for _, r in ticks.iterrows():
        lbl = r.get("label")
        if pd.notna(lbl) and str(lbl) and str(lbl) != "period_end":
            try:
                m = int(float(lbl))
                text = f"{m}'"
            except Exception:
                text = str(lbl)
            ax.text(
                r["x"], -5, text,
                ha="center", va="top",
                fontsize=12,
                color=futicolor.light,
                clip_on=False,
                zorder=4
            )

    # base bars (rounded far end)
    for _, r in mm[mm["type"] == "bar"].iterrows():
        x = float(r["x0"]); w = float(r["x1"] - r["x0"])
        y0, y1 = float(r["y0"]), float(r["y1"])
        if abs(y1 - y0) < 1e-6 or w <= 0:
            continue
        _rounded_end_bar(ax, x=x, w=w, y0=y0, y1=y1, color=r["bar_color"], z=2, alpha=0.5)

    # overlay bars (rounded far end)
    for _, r in mm[mm["type"] == "bar_top"].iterrows():
        x = float(r["x0"]); w = float(r["x1"] - r["x0"])
        y0, y1 = float(r["y0"]), float(r["y1"])
        if abs(y1 - y0) < 1e-6 or w <= 0:
            continue
        _rounded_end_bar(ax, x=x, w=w, y0=y0, y1=y1, color=r["bar_fill"], z=3)

    plt.tight_layout(); plt.show()


def plot_xg_step(df_game: pd.DataFrame, meta: pd.DataFrame, per_bounds: pd.DataFrame) -> None:
    ds = df_match_xg_step(df_game, meta=meta, per_bounds=per_bounds)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=futicolor.dark)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)

    # guides and ticks behind
    for _, r in ds[ds["type"] == "ytick"].iterrows():
        ax.axhline(
            r["y"], 
            lw=3, 
            linestyle=(0, (0.1, 2)),
            dash_capstyle="round", 
            color=futicolor.dark2, 
            clip_on=False,
            zorder=0
            )
        
    # vertical x ticks: only draw HT line; put all labels under the axis
    ticks = ds[ds["type"] == "tick"]

    for _, r in ticks.iterrows():
        if r.get("label") == "period_end":
            y0, y1 = ax.get_ylim()
            ax.vlines(
                r["x"], 
                ymin=-3,
                ymax=y1,
                lw=2, 
                color=futicolor.light, 
                clip_on=False,
                zorder=0
                )  # thick HT line

        lbl = r.get("label")
        if pd.notna(lbl) and str(lbl) and str(lbl) !="period_end":
            ax.text(
                r["x"], -5, str(lbl),
                ha="center", va="top",
                fontsize=12,
                color=futicolor.light,
                clip_on=False,        # let text render outside the axes box
                zorder=4
            ) # text labels for minutes

    # left-side y labels at each horizontal guide, using labels from the dataframe
    yticks = ds.loc[ds["type"] == "ytick"].sort_values(["match_id", "y"])

    for _, r in yticks.iterrows():
        ax.annotate(
            str(r["label"]),
            xy=(0, r["y"]),
            xycoords=ax.get_yaxis_transform(),  # x in axes coords, y in data coords
            xytext=(-6, 0), 
            textcoords="offset points",
            ha="right", 
            va="center",
            fontsize=12, 
            color=futicolor.light,
            clip_on=False, 
            zorder=4
        )


    # step lines by team
    shots = ds[ds["type"] == "shot"]
    for (_, tid), g in shots.groupby(["match_id", "team_id"], sort=False):
        g = g.sort_values("x", kind="stable")
        ax.plot(
            g["x"], 
            g["y"], 
            lw=2.0, 
            color=g["line_color"].iloc[0],
            drawstyle="steps-post", 
            clip_on=False,
            zorder=2
            )

    # goal markers + labels
    goals = ds[ds["type"] == "goal"]
    if not goals.empty:
        ax.scatter(
            goals["x"], 
            goals["y"], 
            s=100, 
            c=goals["point_color"],
            edgecolors=futicolor.dark, 
            linewidths=2, 
            clip_on=False,
            zorder=3
            )

    plt.tight_layout(); plt.show()


def plot_territory(df_game: pd.DataFrame, *, gap: float = 0.8) -> None:
    """
    Draw rounded territory cells with slight transparent gaps so the pitch shows through.
    Pitch logo is off; markings are lifted above the cells.
    """
    tt = df_match_territory(df_game)
    fig, ax, pitch = plot_pitch(orientation="wide", figsize=(10, 7), logo=False, linewidth=2)

    def add_round_rect(x, y, w, h, *, fc, a=1.0, z=3, r_ratio=0.18, r_max=1.2):
        # shrink rectangle to create gaps (transparent → pitch shows)
        w2 = max(0.0, w - gap)
        h2 = max(0.0, h - gap)
        if w2 <= 0.0 or h2 <= 0.0:
            return
        x2 = x + 0.5 * (w - w2)
        y2 = y + 0.5 * (h - h2)

        r = min(w2, h2) * r_ratio
        r = max(0.0, min(r, r_max))
        ax.add_patch(FancyBboxPatch(
            (x2, y2), w2, h2,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor="none",
            linewidth=0, alpha=a, zorder=z, clip_on=True
        ))

    for _, r in tt.iterrows():
        x, y = float(r["x_min"]), float(r["y_min"])
        w, h = float(r["x_max"] - r["x_min"]), float(r["y_max"] - r["y_min"])
        add_round_rect(x, y, w, h, fc=r["fill_color"])

    # lift pitch markings above the cells
    for art in pitch.patches:
        art.set_zorder(10)

    plt.tight_layout(); plt.show()


def plot_action_heatmap(df_game: pd.DataFrame, *, gap: float = 0.4) -> None:
    """
    Combined heatmap (both teams together) using df_match_action_heatmap → both_fill.
    """
    hm = df_match_action_heatmap(df_game)  # returns *_count + *_fill columns
    fig, ax, pitch = plot_pitch(orientation="wide", figsize=(10, 7), logo=False, linewidth=2)

    def add_round_rect(x, y, w, h, *, fc, a, z=3, r_ratio=0.18, r_max=0.9):
        w2 = max(0.0, w - gap)
        h2 = max(0.0, h - gap)
        if w2 <= 0.0 or h2 <= 0.0:
            return
        x2 = x + 0.5 * (w - w2)
        y2 = y + 0.5 * (h - h2)
        r = min(w2, h2) * r_ratio
        r = max(0.0, min(r, r_max))
        ax.add_patch(FancyBboxPatch(
            (x2, y2), w2, h2,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor="none",
            linewidth=0, alpha=a, zorder=z, clip_on=True
        ))

    for _, r in hm.iterrows():
        x, y = float(r["x_min"]), float(r["y_min"])
        w, h = float(r["x_max"] - r["x_min"]), float(r["y_max"] - r["y_min"])
        add_round_rect(x, y, w, h, fc=futicolor.purple, a=float(r["both_fill"]))

    # lift pitch markings above the cells
    for art in pitch.patches:
        art.set_zorder(10)

    plt.tight_layout()
    plt.show()

def plot_action_heatmap_home(df_game: pd.DataFrame, *, gap: float = 0.4) -> None:
    """
    Home-only heatmap using df_match_action_heatmap → home_fill.
    """
    hm = df_match_action_heatmap(df_game)
    fig, ax, pitch = plot_pitch(orientation="wide", figsize=(10, 7), logo=False, linewidth=2)

    def add_round_rect(x, y, w, h, *, fc, a, z=3, r_ratio=0.18, r_max=0.9):
        w2 = max(0.0, w - gap)
        h2 = max(0.0, h - gap)
        if w2 <= 0.0 or h2 <= 0.0:
            return
        x2 = x + 0.5 * (w - w2)
        y2 = y + 0.5 * (h - h2)
        r = min(w2, h2) * r_ratio
        r = max(0.0, min(r, r_max))
        ax.add_patch(FancyBboxPatch(
            (x2, y2), w2, h2,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor="none",
            linewidth=0, alpha=a, zorder=z, clip_on=True
        ))

    for _, r in hm.iterrows():
        x, y = float(r["x_min"]), float(r["y_min"])
        w, h = float(r["x_max"] - r["x_min"]), float(r["y_max"] - r["y_min"])
        add_round_rect(x, y, w, h, fc=futicolor.blue, a=float(r["home_fill"]))

    for art in pitch.patches:
        art.set_zorder(10)

    plt.tight_layout()
    plt.show()


def plot_action_heatmap_away(df_game: pd.DataFrame, *, gap: float = 0.4) -> None:
    """
    Away-only heatmap using df_match_action_heatmap → away_fill.
    """
    hm = df_match_action_heatmap(df_game)
    fig, ax, pitch = plot_pitch(orientation="wide", figsize=(10, 7), logo=False, linewidth=2)

    def add_round_rect(x, y, w, h, *, fc, a, z=3, r_ratio=0.18, r_max=0.9):
        w2 = max(0.0, w - gap)
        h2 = max(0.0, h - gap)
        if w2 <= 0.0 or h2 <= 0.0:
            return
        x2 = x + 0.5 * (w - w2)
        y2 = y + 0.5 * (h - h2)
        r = min(w2, h2) * r_ratio
        r = max(0.0, min(r, r_max))
        ax.add_patch(FancyBboxPatch(
            (x2, y2), w2, h2,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor="none",
            linewidth=0, alpha=a, zorder=z, clip_on=True
        ))

    for _, r in hm.iterrows():
        x, y = float(r["x_min"]), float(r["y_min"])
        w, h = float(r["x_max"] - r["x_min"]), float(r["y_max"] - r["y_min"])
        add_round_rect(x, y, w, h, fc=futicolor.pink, a=float(r["away_fill"]))

    for art in pitch.patches:
        art.set_zorder(10)

    plt.tight_layout()
    plt.show()


def plot_shotmap(df_game: pd.DataFrame) -> None:
    """
    Shot map (wide pitch) with sizes from `shot_size`, face alpha from `alpha`,
    and opaque edges from `edge_color`. No RGBA arrays; no per-type branching.
    """
    sm = df_match_shotmap(df_game) 

    # canvas
    fig, ax, _ = plot_pitch(orientation="wide", figsize=(10, 7), linewidth=2, line_color=futicolor.light1)

    # size mapping: map shot_size ∈ [0,1] to diameter (pt), then to area (pt^2)
    p = np.clip(sm["shot_size"].to_numpy(dtype=float), 0.0, 1.0)
    d_min_pt, d_max_pt = 1.0, 24.0
    d_pt = d_min_pt + (d_max_pt - d_min_pt) * p
    s = (np.pi / 4.0) * (d_pt ** 2)

    # per-shot alpha
    A = np.nan_to_num(sm["alpha"].to_numpy(dtype=float), nan=0.0, posinf=1.0, neginf=0.0)

    # draw faces grouped by unique alpha values (skip 0.0 → no face fill)
    unique_alphas = sorted({float(a) for a in np.round(A, 6)} - {0.0})
    for z, a in enumerate(unique_alphas, start=3):
        m = np.isclose(A, a, rtol=1e-6, atol=1e-6)
        if m.any():
            sub = sm.loc[m]
            ax.scatter(
                sub["x_both"], sub["y_both"],
                s=s[m],
                c=sub["point_color"],   # team fill
                edgecolors="none",
                linewidths=0,
                alpha=a,                # scalar alpha for this group
                zorder=z,
            )

    # use the same linewidth as the point edges for goal trails
    edge_lw = 1.6

    # goal trails (wide frame): white segment from start → end, plus a small white start dot
    m_goal = np.isfinite(sm["xend_both"].to_numpy(dtype=float)) & np.isfinite(sm["yend_both"].to_numpy(dtype=float))
    if m_goal.any():
        x0 = sm.loc[m_goal, "x_both"].to_numpy(dtype=float)
        y0 = sm.loc[m_goal, "y_both"].to_numpy(dtype=float)
        x1 = sm.loc[m_goal, "xend_both"].to_numpy(dtype=float)
        y1 = sm.loc[m_goal, "yend_both"].to_numpy(dtype=float)

        # draw each goal segment behind marker fills
        for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
            ax.plot([xi0, xi1], [yi0, yi1],
                    color=futicolor.light, linewidth=edge_lw, alpha=1.0, zorder=2)

        # small white start dot behind marker fills
        ax.scatter(x0, y0, s=12, c=futicolor.light, edgecolors="none", zorder=2)

    # draw edges once for all points, fully opaque
        ax.scatter(
        sm["x_both"], sm["y_both"],
        s=s,
        facecolors="none",
        edgecolors=sm["edge_color"],
        linewidths=edge_lw,  # was 1.6
        zorder=10,
    )

    plt.tight_layout()
    plt.show()


def plot_shotmap_home(df_game: pd.DataFrame) -> None:
    """
    Home team shots only, top-half pitch.
    Uses: x_team,y_team (already in team-attacking-top frame),
          point_color, edge_color, alpha, shot_size from df_match_shotmap.
    """
    sm = df_match_shotmap(df_game)
    sm = sm.loc[sm["home"].astype(str).str.lower().eq("home")].copy()
    if sm.empty:
        return

    # top half canvas (no logo)
    fig, ax, _ = plot_pitch(
        orientation="top", figsize=(7, 5),
        linewidth=2, line_color=futicolor.light1, logo=False
    )

    # marker sizes from shot_size ∈ [0,1]
    p = np.clip(sm["shot_size"].to_numpy(dtype=float), 0.0, 1.0)
    d_min_pt, d_max_pt = 1.0, 24.0
    d_pt = d_min_pt + (d_max_pt - d_min_pt) * p
    s = (np.pi / 4.0) * (d_pt ** 2)

    # fill by unique alpha (skip 0.0), then opaque edges over all points
    A = np.nan_to_num(sm["alpha"].to_numpy(dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    unique_alphas = sorted({float(a) for a in np.round(A, 6)} - {0.0})
    for z, a in enumerate(unique_alphas, start=3):
        m = np.isclose(A, a, rtol=1e-6, atol=1e-6)
        if m.any():
            sub = sm.loc[m]
            ax.scatter(
                sub["x_team"], sub["y_team"],
                s=s[m],
                c=sub["point_color"],
                edgecolors="none",
                linewidths=0,
                alpha=a,
                zorder=z,
            )

    # use the same linewidth as the point edges for goal trails
    edge_lw = 1.6

    # goal trails (team frame): white segment from start → end, plus a small white start dot
    m_goal = np.isfinite(sm["xend_team"].to_numpy(dtype=float)) & np.isfinite(sm["yend_team"].to_numpy(dtype=float))
    if m_goal.any():
        x0 = sm.loc[m_goal, "x_team"].to_numpy(dtype=float)
        y0 = sm.loc[m_goal, "y_team"].to_numpy(dtype=float)
        x1 = sm.loc[m_goal, "xend_team"].to_numpy(dtype=float)
        y1 = sm.loc[m_goal, "yend_team"].to_numpy(dtype=float)

        for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
            ax.plot([xi0, xi1], [yi0, yi1],
                    color=futicolor.light, linewidth=edge_lw, alpha=1.0, zorder=2)

        ax.scatter(x0, y0, s=12, c=futicolor.light, edgecolors="none", zorder=2)

    ax.scatter(
        sm["x_team"], sm["y_team"],
        s=s,
        facecolors="none",
        edgecolors=sm["edge_color"],
        linewidths=edge_lw,
        zorder=10,
    )

    plt.tight_layout()
    plt.show()


def plot_shotmap_away(df_game: pd.DataFrame) -> None:
    """
    Away team shots only, bottom-half pitch.
    Same rendering as home; just switch to orientation="bottom".
    """
    sm = df_match_shotmap(df_game)
    sm = sm.loc[sm["home"].astype(str).str.lower().eq("away")].copy()
    if sm.empty:
        return

    fig, ax, _ = plot_pitch(
        orientation="top", figsize=(7, 5),
        linewidth=2, line_color=futicolor.light1, logo=False
    )

    p = np.clip(sm["shot_size"].to_numpy(dtype=float), 0.0, 1.0)
    d_min_pt, d_max_pt = 1.0, 24.0
    d_pt = d_min_pt + (d_max_pt - d_min_pt) * p
    s = (np.pi / 4.0) * (d_pt ** 2)

    A = np.nan_to_num(sm["alpha"].to_numpy(dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    unique_alphas = sorted({float(a) for a in np.round(A, 6)} - {0.0})
    for z, a in enumerate(unique_alphas, start=3):
        m = np.isclose(A, a, rtol=1e-6, atol=1e-6)
        if m.any():
            sub = sm.loc[m]
            ax.scatter(
                sub["x_team"], sub["y_team"],
                s=s[m],
                c=sub["point_color"],
                edgecolors="none",
                linewidths=0,
                alpha=a,
                zorder=z,
            )

    # use the same linewidth as the point edges for goal trails
    edge_lw = 1.6  # replace the later scatter(..., linewidths=1.6, ...) with linewidths=edge_lw

    # goal trails (team frame): white segment from start → end, plus a small white start dot
    m_goal = np.isfinite(sm["xend_team"].to_numpy(dtype=float)) & np.isfinite(sm["yend_team"].to_numpy(dtype=float))
    if m_goal.any():
        x0 = sm.loc[m_goal, "x_team"].to_numpy(dtype=float)
        y0 = sm.loc[m_goal, "y_team"].to_numpy(dtype=float)
        x1 = sm.loc[m_goal, "xend_team"].to_numpy(dtype=float)
        y1 = sm.loc[m_goal, "yend_team"].to_numpy(dtype=float)

        for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
            ax.plot([xi0, xi1], [yi0, yi1],
                    color=futicolor.light, linewidth=edge_lw, alpha=1.0, zorder=2)

        ax.scatter(x0, y0, s=12, c=futicolor.light, edgecolors="none", zorder=2)

    ax.scatter(
        sm["x_team"], sm["y_team"],
        s=s,
        facecolors="none",
        edgecolors=sm["edge_color"],
        linewidths=edge_lw,
        zorder=10,
    )

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

#%%
df = load_sample_data("vaep")

def main():
    match_id = pick_game(df)
    df_game = df[df["match_id"] == match_id].copy()

    meta, per_bounds = compute_timing_meta(df_game, bin_seconds=BIN, plot_gap_seconds=0.7 * BIN)

    plot_momentum(df_game, meta)
    plot_xg_step(df_game, meta, per_bounds)
    plot_territory(df_game)
    plot_action_heatmap(df_game)          # combined (purple)
    plot_action_heatmap_home(df_game)     # home-only (blue)
    plot_action_heatmap_away(df_game)     # away-only (pink)
    plot_shotmap(df_game)
    plot_shotmap_home(df_game)
    plot_shotmap_away(df_game)


if __name__ == "__main__":
    main()

# %%
