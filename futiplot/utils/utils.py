import os
import pandas as pd
import numpy as np
from importlib.resources import files
from typing import Optional, Any

def load_sample_data(data_name: str) -> pd.DataFrame:
    """
    Load a sample CSV bundled in futiplot/sample_data as a DataFrame.

    Looks for a file named 'sample_<data_name>.csv'.
    """
    resource = files("futiplot").joinpath("sample_data", f"sample_{data_name}.csv")
    if not resource.is_file():
        raise FileNotFoundError(f"Sample data file '{data_name}' not found at {resource!s}")
    with resource.open("rb") as f:
        return pd.read_csv(f)


def transform_xy(
    data: pd.DataFrame,
    pitch: object | dict | None = None,
    flip_coords: bool = False,
) -> pd.DataFrame:
    """
    convert raw event coords to plotting coords.
    defaults: length=105, width=68, orientation='tall'.

    semantics (unchanged):
      - flip_coords=True mirrors the length axis (away team).
      - orientation='wide': plot x=length, y=width; keep home bottom->top by inverting width for home only.
      - orientation='tall': rotate so x=width, y=length; mirror width for away so home still reads bottom->top.
    """
    # resolve pitch params without importing PlotPitch
    if pitch is None:
        L, W, orient = 105.0, 68.0, "tall"
    elif isinstance(pitch, dict):
        L = float(pitch.get("pitch_length", 105.0))
        W = float(pitch.get("pitch_width", 68.0))
        orient = str(pitch.get("orientation", "tall"))
    else:
        L = float(getattr(pitch, "pitch_length", 105.0))
        W = float(getattr(pitch, "pitch_width", 68.0))
        orient = str(getattr(pitch, "orientation", "tall"))

    out = data.copy()

    # use raw arrays once; formulas don’t mutate inputs
    x1 = data["x_start"].to_numpy(dtype=float, copy=False)
    y1 = data["y_start"].to_numpy(dtype=float, copy=False)
    has_end = {"x_end", "y_end"}.issubset(data.columns)
    if has_end:
        x2 = data["x_end"].to_numpy(dtype=float, copy=False)
        y2 = data["y_end"].to_numpy(dtype=float, copy=False)

    if orient == "wide":
        # home: (x, y) = (x, W - y)
        # away: (x, y) = (L - x, y)
        out["x_start"] = np.where(flip_coords, L - x1, x1)
        out["y_start"] = np.where(flip_coords, y1, W - y1)
        if has_end:
            out["x_end"] = np.where(flip_coords, L - x2, x2)
            out["y_end"] = np.where(flip_coords, y2, W - y2)
    else:
        # tall (rotated):
        # home: (x, y) = (y, x)
        # away: (x, y) = (W - y, L - x)
        out["x_start"] = np.where(flip_coords, W - y1, y1)
        out["y_start"] = np.where(flip_coords, L - x1, x1)
        if has_end:
            out["x_end"] = np.where(flip_coords, W - y2, y2)
            out["y_end"] = np.where(flip_coords, L - x2, x2)

    return out



def get_zones(pitch: Optional[Any] = None, x_zones=None, y_zones=None):
    """
    compute zone breakpoints for a pitch.

    if pitch is None, defaults to length=105, width=68, orientation='wide'.
    accepts a PlotPitch-like object (attrs) or a dict with keys
    {'pitch_length','pitch_width','orientation'}.

    returns: length_zones, width_zones (np.ndarray each)
    """
    # resolve pitch params
    if pitch is None:
        L, W, orient = 105.0, 68.0, "wide"
    elif isinstance(pitch, dict):
        L  = float(pitch.get("pitch_length", 105.0))
        W  = float(pitch.get("pitch_width", 68.0))
        orient = str(pitch.get("orientation", "wide"))
    else:
        L  = float(getattr(pitch, "pitch_length", 105.0))
        W  = float(getattr(pitch, "pitch_width", 68.0))
        orient = str(getattr(pitch, "orientation", "wide"))

    # helper to turn "zones" into edges
    def calc(z, span, defaults):
        if z is None:
            return np.asarray(defaults, dtype=float)
        if isinstance(z, int):
            return np.linspace(0.0, span, z + 1, dtype=float)
        if isinstance(z, (list, tuple, np.ndarray)):
            edges = [0.0, *z, span]
            # unique + sorted, preserve endpoints
            edges = sorted(set(float(v) for v in edges))
            return np.asarray(edges, dtype=float)
        raise ValueError("zones must be an int, a list/tuple of breakpoints, or None")

    # defaults (expressed off resolved L/W)
    default_x = [0.0, 16.5, L/3.0, L/2.0, 2*L/3.0, L-16.5, L]
    default_y = [0.0, W/2.0-20.16, W/2.0-9.16, W/2.0+9.16, W/2.0+20.16, W]

    # map according to orientation
    if orient == "tall":
        length_zones = calc(y_zones, W, default_y)  # raw width runs along pitch length
        width_zones  = calc(x_zones, L, default_x)  # raw length runs along pitch width
    else:  # 'wide'
        length_zones = calc(x_zones, L, default_x)
        width_zones  = calc(y_zones, W, default_y)

    return length_zones, width_zones