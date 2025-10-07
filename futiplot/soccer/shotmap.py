import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .pitch import plot_pitch
from ..utils.colors import futicolor
from ..utils.utils import transform_xy

def plot_shotmap(
    shots: pd.DataFrame,
    pitch,
    figsize=(6.5, 4),
    goal_color=futicolor.green,  # Default fill color for goals
    miss_color=futicolor.dark2,   # Default fill color for misses
    line_color=futicolor.light,  # Default line color for pitch & goal lines
    size_range=(100, 1000),       # (min_size, max_size) for dot sizes
    pitch_line_color=futicolor.light1,
):
    """
    Plot a shot map on a zoomed-in pitch area near the goal.

    Parameters
    ----------
    shots : pd.DataFrame
        Must include columns:
          - 'x_start', 'y_start': shot coordinates
          - 'xg'               : expected goals value
          - 'goal'             : boolean or int/str representing whether it was scored
    pitch : PlotPitch
        A futiplot pitch object that defines pitch_length, pitch_width, orientation, etc.
    figsize : tuple
        Width and height in inches for the figure (default=(10, 7)).
    shot_color : str
        The fill color for the shot circles (default=futicolor.green).
    line_color : str
        The line color for pitch markings and goal lines (default=futicolor.light).
    size_range : tuple
        (min_size, max_size) for the circle sizes based on xG (default=(100, 1000)).

    Returns
    -------
    (fig, ax) : tuple
        The Matplotlib figure and axis objects containing the shot map.
    """

    # --- Step 1: Validate and clean data ---
    required_cols = {"x_start", "y_start", "xg", "goal"}
    if not required_cols.issubset(shots.columns):
        raise ValueError(f"Shots DataFrame must have columns: {required_cols}")

    # Drop rows with NaN in critical columns
    shots = shots.dropna(subset=required_cols)

    # Convert goal column to boolean (handles 0/1, 'TRUE'/'FALSE', etc.)
    shots["goal"] = shots["goal"].astype(bool)

    # Ensure size_range is valid
    if not (
        isinstance(size_range, tuple) 
        and len(size_range) == 2 
        and size_range[0] < size_range[1]
    ):
        raise ValueError("size_range must be (min_size, max_size) with min_size < max_size.")

    min_size, max_size = size_range

    # --- Step 2: Transform and sort shots so smaller circles go on top ---
    transformed_shots = transform_xy(shots, pitch)
    # Sort by xg so smaller circles are drawn last (on top)
    transformed_shots.sort_values(by="xg", ascending=False, inplace=True)

    # --- Step 3: Create the pitch figure ---
    fig, ax, pitch = plot_pitch(
        figsize=figsize,
        pitch_length=pitch.pitch_length,
        pitch_width=pitch.pitch_width,
        orientation=pitch.orientation,
        pitch_color="none",  # Transparent pitch background
        line_color=pitch_line_color,
    )
    fig.patch.set_alpha(0)  # Transparent figure background

    # --- Step 4: Dynamically zoom into the vertical area around the goal ---
    buffer = 5  # Extra space around the zoomed area

    # Select the correct column for vertical coordinates based on pitch orientation:
    # - For a tall pitch: y_start is vertical.
    # - For a wide pitch: x_start serves as the vertical coordinate.
    if pitch.orientation == "tall":
        vertical_coords = transformed_shots["y_start"]
    else:
        vertical_coords = transformed_shots["x_start"]

    # Compute the minimum and maximum vertical coordinate from the shots.
    min_shot_vert = vertical_coords.min()
    max_shot_vert = vertical_coords.max()

    if pitch.orientation == "tall":
        # For a tall pitch, the default vertical limits show the attacking half:
        # Bottom of the view is half the pitch length and the top extends to the end plus a buffer.
        default_bottom = pitch.pitch_length * 0.498
        default_top = pitch.pitch_length + buffer

        # Adjust vertical limits if any shot lies outside the default view.
        bottom_limit = min(default_bottom, min_shot_vert - buffer)
        top_limit = max(default_top, max_shot_vert + buffer)
        
        # Apply the adjusted vertical limits.
        ax.set_ylim(bottom_limit, top_limit)
        
        # Always show the full width of the pitch horizontally.
        ax.set_xlim(0 - buffer, pitch.pitch_width + buffer)

    else:
        # For a wide pitch, the default vertical limits are based on the pitch length.
        # Here, the view is fixed vertically from 0 to the full pitch length.
        default_bottom = 0
        default_top = pitch.pitch_length

        # Adjust vertical limits if any shot lies outside this default range.
        bottom_limit = min(default_bottom, min_shot_vert - buffer)
        top_limit = max(default_top, max_shot_vert + buffer)
        
        # Apply the adjusted vertical limits.
        ax.set_ylim(bottom_limit, top_limit)
        
        # Horizontally, always display the entire pitch width.
        ax.set_xlim(0 - buffer, pitch.pitch_width + buffer)


    # --- Step 5: Find the goal center (orientation-dependent) ---
    # Adjust if your coordinate system differs (e.g., 0 vs pitch_length).
    if pitch.orientation == "tall":
        # Potentially top center of the pitch
        goal_coords = pd.DataFrame([
            {"x_start": pitch.pitch_length, "y_start": pitch.pitch_width / 2}
        ])
    else:
        # Potentially right center of the pitch
        goal_coords = pd.DataFrame([
            {"x_start": pitch.pitch_length, "y_start": pitch.pitch_width / 2}
        ])
    goal_center_x, goal_center_y = transform_xy(goal_coords, pitch).iloc[0]

    # --- Step 6: Precompute edge colors & line widths in one pass ---
    # This way, we only call scatter() once.
    edge_colors = transformed_shots["goal"].apply(
        lambda g: line_color if g else futicolor.dark
    )
    shot_colors = transformed_shots["goal"].apply(
        lambda g: goal_color if g else miss_color
    )
    alphas = transformed_shots["goal"].apply(
        lambda g: 1 if g else 1
    )
    line_widths = transformed_shots["goal"].apply(
        lambda g: 1.5 if g else 1
    )

    # Helper: clamp xG-based sizes to [min_size, max_size]
    def scale_xg(xg_series):
        return np.clip(xg_series * max_size, min_size, max_size)

    # --- Step 7: Scatter plot all shots in one go ---
    ax.scatter(
        x=transformed_shots["x_start"],
        y=transformed_shots["y_start"],
        s=scale_xg(transformed_shots["xg"]),
        c=shot_colors,               # Fill color
        alpha=alphas,
        edgecolors=edge_colors,     # Goal => line_color, Miss => futicolor.dark
        linewidths=line_widths,     # Thicker edge if goal
        zorder=3
    )

    # --- Step 8: Draw goal lines only for actual goals ---
    # Now that all circles are drawn in ascending xG order,
    # smaller circles (low xG) will appear on top of bigger circles (high xG).
    scored_shots = transformed_shots[transformed_shots["goal"]]
    for _, row in scored_shots.iterrows():
        ax.plot(
            [row["x_start"], goal_center_x],
            [row["y_start"], goal_center_y],
            color=line_color,
            linewidth=1.5,
            zorder=2
        )

    return fig, ax
