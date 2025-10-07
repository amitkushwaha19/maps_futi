# df_season_territory.py

import numpy as np
import pandas as pd
from typing import Union

from futiplot.utils.utils import transform_xy
from futiplot.utils import futicolor


def df_season_territory(df: pd.DataFrame, team_id: Union[int, str]) -> pd.DataFrame:
    """
    season-level territory bins for one team (team vs opponents), in the team's frame.

    returns a dataframe with one row per spatial bin containing:
        team_id, x_min, x_max, y_min, y_max, team_label, possession_label, fill_color

    notes on frames:
      - we first put each match into a common "home frame" using transform_xy
        (home: no flip, away: length flip).
      - then we make it team-centric: for matches where the given team played away,
        we flip both axes to mirror that game's already-home-framed coords so that
        the team always attacks left→right and bottom→top consistently.

    color logic:
      - if the team has >=50% of touches in the bin, label is the team and color is blue
        (≤55% → blue1, >55% → blue). otherwise label is "opponents" and color is pink
        (≤55% → pink1, >55% → pink).
    """
    # required columns (we use these unconditionally)
    need = {
        "match_id", "team_id", "home",
        "x_start", "y_start",
        "type_name", "phase_team"
    }
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame(
            columns=[
                "team_id", "x_min", "x_max", "y_min", "y_max",
                "team_label", "possession_label", "fill_color"
            ]
        )

    xcol, ycol = "x_start", "y_start"

    # keep only in-possession actions and drop restarts
    exclude = {"corner_crossed", "corner_short", "throw_in", "goal_kick"}
    df = df.loc[(df["team_id"] == df["phase_team"]) & (~df["type_name"].isin(exclude))].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "team_id", "x_min", "x_max", "y_min", "y_max",
                "team_label", "possession_label", "fill_color"
            ]
        )

    # fixed bin edges (wide orientation)
    length_zones = np.array([0.0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0], dtype=float)
    width_zones  = np.array([0.0, 13.84, 24.84, 43.16, 54.16, 68.0],   dtype=float)
    nx, ny = len(length_zones) - 1, len(width_zones) - 1

    # put each match into a common "home frame"
    pitch = {"pitch_length": 105.0, "pitch_width": 68.0, "orientation": "wide"}
    keep  = ["match_id", "team_id", "home", xcol, ycol]
    home  = transform_xy(df[df["home"] == "home"][keep], pitch=pitch, flip_coords=False)
    away  = transform_xy(df[df["home"] == "away"][keep], pitch=pitch, flip_coords=True)
    both  = pd.concat([home, away], ignore_index=True)

    # convert to a team-centric frame: flip BOTH axes for matches where our team was away
    L, W = float(pitch["pitch_length"]), float(pitch["pitch_width"])
    was_away = set(
        df.loc[(df["team_id"] == team_id) & (df["home"] == "away"), "match_id"].dropna().unique()
    )
    if was_away:
        m = both["match_id"].isin(was_away)
        both.loc[m, xcol] = L - both.loc[m, xcol]
        both.loc[m, ycol] = W - both.loc[m, ycol]

    # mark rows as team vs opponent
    both["side"] = np.where(both["team_id"] == team_id, "team", "opponent")

    # vectorized binning
    xb = np.searchsorted(length_zones, both[xcol].to_numpy(), side="right") - 1
    yb = np.searchsorted(width_zones,  both[ycol].to_numpy(),  side="right") - 1
    np.clip(xb, 0, nx - 1, out=xb)
    np.clip(yb, 0, ny - 1, out=yb)
    both["x_bin"] = xb.astype(np.int16)
    both["y_bin"] = yb.astype(np.int16)

    # counts across the season in each bin
    cnt = (
        both.groupby(["x_bin", "y_bin", "side"], observed=True)
            .size()
            .unstack("side", fill_value=0)
            .rename(columns={"team": "touches_team", "opponent": "touches_opp"})
            .reset_index()
    )
    tot = cnt["touches_team"] + cnt["touches_opp"]
    cnt["poss_team_pct"] = np.where(tot > 0, cnt["touches_team"] / tot, 0.5)

    # rectangle coords from bin edges
    xb = cnt["x_bin"].to_numpy()
    yb = cnt["y_bin"].to_numpy()
    cnt["x_min"] = length_zones[xb]
    cnt["x_max"] = length_zones[xb + 1]
    cnt["y_min"] = width_zones[yb]
    cnt["y_max"] = width_zones[yb + 1]

    # pick a display name for team_id if present; else label as "team"
    if "team_name" in df.columns:
        cand = df.loc[df["team_id"] == team_id, "team_name"].dropna()
        team_name = cand.iloc[0] if not cand.empty else "team"
    else:
        team_name = "team"

    # labels and colors
    team_dom        = cnt["poss_team_pct"] >= 0.50
    poss_for_label  = np.where(team_dom, cnt["poss_team_pct"], 1.0 - cnt["poss_team_pct"])
    cnt["team_label"] = np.where(team_dom, team_name, "opponents")
    cnt["possession_label"] = (np.round(poss_for_label * 100).astype(int)).astype(str) + "%"

    cnt["fill_color"] = np.where(
        team_dom,
        np.where(poss_for_label <= 0.55, futicolor.blue1, futicolor.blue),
        np.where(poss_for_label <= 0.55, futicolor.pink1, futicolor.pink),
    )

    # final shape for plotting
    out = (
        cnt.assign(team_id=team_id)[
            ["team_id", "x_min", "x_max", "y_min", "y_max",
             "team_label", "possession_label", "fill_color"]
        ]
        .sort_values(["y_min", "x_min"], kind="stable")
        .reset_index(drop=True)
    )
    return out
