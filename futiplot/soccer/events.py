# events.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..utils.utils       import transform_xy
from ..utils.colors      import futicolor
from ..utils.plotting    import plot_comet, plot_dotted_line, plot_point

def plot_ball_kicked(
    data,
    pitch,
    ax,
    event_color=futicolor.green,
    linewidth=1.5,
    curvature=0.0,
    segments_per_unit=20,
    flip_coords=False,
):
    """
    Plots ball-kicked events (e.g., passes, shots) with a comet tail effect.

    Parameters:
        data: pandas DataFrame containing event data with columns
            'x_start', 'y_start', 'x_end', 'y_end', 'type_name'.
        pitch: PlotPitch instance defining pitch dimensions and appearance.
        ax: matplotlib Axes object on which to draw.
        event_color: Single color or list of hex codes for gradient tails.
        linewidth: Maximum width of each tail at its tip.
        curvature: Float between 0.0 (straight) and 1.0 (pronounced curve).
        segments_per_unit: Number of segments per unit length for comet tails.
        flip_coords: Whether to flip x/y coordinates for orientation.
    """
    transformed_data = transform_xy(data, pitch=pitch, flip_coords=flip_coords)

    ball_kicked = transformed_data[
        transformed_data["type_name"].isin([
            "pass", "cross", "shot", "shot_penalty", "clearance",
            "corner_short", "corner_crossed", "freekick", "freekick_short",
            "freekick_crossed", "throw_in", "goalkick"
        ])
    ]

    start_coords = ball_kicked[["x_start", "y_start"]].to_numpy()
    end_coords = ball_kicked[["x_end", "y_end"]].to_numpy()

    plot_comet(
        ax,
        start_coords,
        end_coords,
        pitch_color=getattr(pitch, "pitch_color", futicolor.dark),
        event_color=event_color,
        linewidth=linewidth,
        curvature=curvature,
        segments_per_unit=segments_per_unit,
        zorder=1,
    )


def plot_ball_carried(
    data,
    pitch,
    ax,
    event_color=futicolor.green,
    linewidth=1.5,
    dot_spacing=0.8,
    flip_coords=False,
):
    """
    Plots ball-carried events (e.g., dribbles) as dotted lines.

    Parameters:
        data: pandas DataFrame containing event data with 'x_start', 'y_start', 'x_end', 'y_end', 'type_name'.
        pitch: PlotPitch instance defining pitch dimensions and appearance.
        ax: matplotlib Axes object on which to draw.
        event_color: Color of the dotted lines.
        linewidth: Width factor for determining dot size.
        dot_spacing: Spacing between dots along each dribble.
        flip_coords: Whether to flip x/y coordinates for orientation.
    """
    transformed_data = transform_xy(data, pitch=pitch, flip_coords=flip_coords)

    ball_carried = transformed_data[
        transformed_data["type_name"] == "dribble"
    ]

    start_coords = ball_carried[["x_start", "y_start"]].to_numpy()
    end_coords = ball_carried[["x_end", "y_end"]].to_numpy()

    plot_dotted_line(
        ax,
        start_coords,
        end_coords,
        color=event_color,
        dot_size=linewidth / 2,
        dot_spacing=dot_spacing,
        zorder=2,
    )


def plot_ball_static(
    data,
    pitch,
    ax,
    event_color=futicolor.green,
    event_size=100,
    flip_coords=False,
):
    """
    Plots static events (e.g., tackles, saves) as points.

    Parameters:
        data: pandas DataFrame containing event data with 'x_start', 'y_start', 'type_name'.
        pitch: PlotPitch instance defining pitch dimensions and appearance.
        ax: matplotlib Axes object on which to draw.
        event_color: Color of event markers.
        event_size: Size of event markers.
        flip_coords: Whether to flip x/y coordinates for orientation.
    """
    transformed_data = transform_xy(data, pitch=pitch, flip_coords=flip_coords)

    ball_static = transformed_data[
        transformed_data["type_name"].notna() &
        ~transformed_data["type_name"].isin([
            "pass", "cross", "shot", "shot_penalty", "clearance",
            "corner", "corner_crossed", "freekick", "freekick_short",
            "freekick_crossed", "throw_in", "goalkick", "dribble"
        ])
    ]

    static_coords = ball_static[["x_start", "y_start"]].to_numpy()

    plot_point(
        ax,
        static_coords,
        color=event_color,
        size=event_size,
        edgecolor=getattr(pitch, "pitch_color", futicolor.dark),
        linewidth=1.5,
        zorder=3,
    )


def plot_events(
    data,
    pitch,
    ax=None,
    event_color=futicolor.green,
    event_size=100,
    linewidth=5,
    curvature=0.0,
    segments_per_unit=20,
    dot_spacing=0.8,
    flip_coords=False,
):
    """
    Plots all events (kicked, carried, and static) on the given axis.

    Parameters:
        data: pandas DataFrame containing event data.
        pitch: PlotPitch instance defining pitch dimensions and appearance.
        ax: matplotlib Axes object to draw on; if None, a new figure and axis are created.
        event_color: Single color or list of hex codes for gradients and markers.
        event_size: Size of static event markers.
        linewidth: Width of line-based events at their tips.
        curvature: Float between 0.0 and 1.0 controlling arc of comet tails.
        segments_per_unit: Number of segments per unit length for comet tails.
        dot_spacing: Spacing between dots for carried events.
        flip_coords: Whether to flip x/y coordinates for orientation.
    Returns:
        If ax was None, returns (fig, ax); otherwise returns ax.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    else:
        fig = None

    plot_ball_kicked(
        data,
        pitch,
        ax,
        event_color=event_color,
        linewidth=linewidth,
        curvature=curvature,
        segments_per_unit=segments_per_unit,
        flip_coords=flip_coords,
    )

    plot_ball_carried(
        data,
        pitch,
        ax,
        event_color=event_color,
        linewidth=linewidth,
        dot_spacing=dot_spacing,
        flip_coords=flip_coords,
    )

    plot_ball_static(
        data,
        pitch,
        ax,
        event_color=event_color,
        event_size=event_size,
        flip_coords=flip_coords,
    )

    if fig is not None:
        return fig, ax
    return ax


def plot_passage(
    data,
    pitch,
    ax,
    event_size=120,
    linewidth=5,
    curvature=0.0,
    segments_per_unit=30,
    dot_spacing=0.8,
):
    """
    Plots events for both teams on the same pitch.

    Parameters:
        data: pandas DataFrame with events from both teams; must include a 'home' column.
        pitch: PlotPitch instance defining pitch dimensions and appearance.
        ax: matplotlib Axes object to draw on.
        event_size: Size of static event markers.
        linewidth: Width of line-based events at their tips.
        curvature: Float between 0.0 and 1.0 controlling arc of comet tails.
        segments_per_unit: Number of segments per unit length for comet tails.
        dot_spacing: Spacing between dots for carried events.
    """
    home_events = data[data["home"] == "home"]
    away_events = data[data["home"] == "away"]

    plot_events(
        home_events,
        pitch,
        ax=ax,
        event_color=futicolor.blue,
        event_size=event_size,
        linewidth=linewidth,
        curvature=curvature,
        segments_per_unit=segments_per_unit,
        dot_spacing=dot_spacing,
        flip_coords=False,
    )

    plot_events(
        away_events,
        pitch,
        ax=ax,
        event_color=futicolor.pink,
        event_size=event_size,
        linewidth=linewidth,
        curvature=curvature,
        segments_per_unit=segments_per_unit,
        dot_spacing=dot_spacing,
        flip_coords=True,
    )
