#%%
# df_player_season_touch_heatmap.py

import numpy as np
import pandas as pd
from typing import Tuple
from scipy.ndimage import gaussian_filter

from futiplot.utils.utils import get_zones
from futiplot.utils import futicolor


def df_season_player_touch_heatmap(
    df: pd.DataFrame,
    player_id: int | str,
    x_zones: int = 21,
    y_zones: int = 14,
    blur: float | None = None,
    color_range: Tuple[str | float, str | float] = ("5%", "95%"),
) -> pd.DataFrame:
    """
    build a season-level touch heatmap dataframe for one player.

    uses every touch with this player_id, puts all matches into a
    player-centric frame (his team always attacks the same direction),
    bins into 21×14, applies a default gaussian blur, then scales intensity
    to an alpha in [0,1] using a percentile window.

    returns one row per zone with:
      player_id, x_min, x_max, y_min, y_max, count, intensity, alpha
    """
    need = {"match_id", "team_id", "home", "player_id", "x_start", "y_start"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(columns=[
            "player_id", "x_min", "x_max", "y_min", "y_max", "count", "intensity", "alpha"
        ])

    # keep only this player's touches
    touch = df.loc[
        (df["player_id"] == player_id),
        ["match_id", "team_id", "home", "x_start", "y_start"],
    ].dropna(subset=["x_start", "y_start"])
    if touch.empty:
        return pd.DataFrame(columns=[
            "player_id", "x_min", "x_max", "y_min", "y_max", "count", "intensity", "alpha"
        ])

    # pitch zones
    length_zones, width_zones = get_zones(None, x_zones, y_zones)
    nx, ny = len(length_zones) - 1, len(width_zones) - 1
    W = float(68.0)

    # transform to plotting frame: invert y (bottom→top). (x left as-is here)
    x = touch["x_start"].to_numpy()
    y = (W - touch["y_start"].to_numpy())

    # 2d histogram (x along length, y along width); transpose to (ny, nx)
    H, _, _ = np.histogram2d(x, y, bins=[length_zones, width_zones])
    counts = H.T  # (ny, nx)

    # blur (default ≈ min(nx, ny) / 10)
    if blur is None:
        blur = max(1.0, min(nx, ny) / 10.0)
    intensity = gaussian_filter(counts, sigma=blur) if blur > 0 else counts

    # percentile window → alpha in [0,1]
    vals = intensity[intensity > 0]
    vmin, vmax = color_range
    if isinstance(vmin, str) and vmin.endswith("%"):
        vmin = np.percentile(vals, float(vmin[:-1])) if vals.size else 0.0
    if isinstance(vmax, str) and vmax.endswith("%"):
        vmax = np.percentile(vals, float(vmax[:-1])) if vals.size else 1.0
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1.0

    alpha = np.clip((intensity - vmin) / (vmax - vmin), 0.0, 1.0)

    # rectangle edges matching (ny, nx) layout
    X_min, Y_min = np.meshgrid(length_zones[:-1], width_zones[:-1], indexing="xy")
    X_max, Y_max = np.meshgrid(length_zones[1:],  width_zones[1:],  indexing="xy")

    out = (
        pd.DataFrame(
            {
                "player_id": player_id,
                "x_min": X_min.ravel().astype(float),
                "x_max": X_max.ravel().astype(float),
                "y_min": Y_min.ravel().astype(float),
                "y_max": Y_max.ravel().astype(float),
                "alpha": alpha.ravel().astype(float),
            }
        )
        .sort_values(["y_min", "x_min"], kind="stable")
        .reset_index(drop=True)
    )
    return out
