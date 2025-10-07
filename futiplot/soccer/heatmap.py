import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
from matplotlib.collections import PatchCollection
from matplotlib.patheffects import withStroke
from typing import Optional, Union, List, Tuple
from scipy.ndimage import gaussian_filter
from ..utils.colors import futicolor
from ..utils.utils import transform_xy, get_zones
from .pitch import PlotPitch

def _overlay_pitch_markings(ax: plt.Axes, pitch: PlotPitch):
    """
    Draw a transparent overlay of the pitch on ax, inheriting buffer.
    """
    overlay = PlotPitch(
        pitch_length=pitch.pitch_length,
        pitch_width=pitch.pitch_width,
        orientation=pitch.orientation,
        pitch_color="none",
        line_color=pitch.line_color,
        linewidth=pitch.linewidth,
        spot_radius=pitch.spot_radius,
        buffer=pitch.buffer,
    )
    overlay.construct_pitch()
    overlay.draw(ax)


def get_heatmap(transformed_data, length_zones, width_zones, aggregation, value_column):
    """
    Generate a heatmap by aggregating event data into defined zones.

    Parameters:
        transformed_data (pd.DataFrame): Event data with 'x_start', 'y_start', 'x_end', 'y_end', or value_column.
        length_zones (array): Zone edges along pitch length.
        width_zones (array): Zone edges along pitch width.
        aggregation (str): Aggregation function (e.g., "count", "sum", "mean", "std", "sum_sq").
        value_column (str): Column to aggregate if applicable.

    Returns:
        np.ndarray: 2D heatmap array with aggregated values.

    Example:
        heatmap = get_heatmap(data, pitch, length_zones, width_zones, aggregation="count")
    """
    transformed_data=transformed_data.copy()
    
    # Determine x and y columns for event bins
    x_col, y_col = ("x_start", "y_start")

    # Bin data into x and y zones
    transformed_data["x_bin"] = pd.cut(
        transformed_data[x_col], bins=length_zones, labels=False, include_lowest=True
    ).clip(0, len(length_zones) - 2)  # Clip to avoid out-of-bounds indices
    transformed_data["y_bin"] = pd.cut(
        transformed_data[y_col], bins=width_zones, labels=False, include_lowest=True
    ).clip(0, len(width_zones) - 2)  # Clip to avoid out-of-bounds indices

    # Initialize heatmap array
    heatmap = np.zeros((len(width_zones) - 1, len(length_zones) - 1))

    # Group data by bins and aggregate based on chosen method
    grouped = transformed_data.groupby(["x_bin", "y_bin"])
    total_events = len(transformed_data) if aggregation == "proportion" else None

    for (x_bin, y_bin), group in grouped:
        x_bin = int(x_bin)
        y_bin = int(y_bin)

        # Calculate aggregation value for the bin
        if aggregation == "count":
            value = len(group)
        elif aggregation == "sum":
            value = group[value_column].sum()
        elif aggregation == "mean":
            value = group[value_column].mean()
        elif aggregation == "std":
            value = group[value_column].std()
        elif aggregation == "sum_sq":  # Sum of squares for variance calculation
            value = (group[value_column] ** 2).sum()
        elif aggregation == "proportion":
            value = len(group) / total_events if total_events else 0
        else:
            raise ValueError(f"unsupported aggregation method: {aggregation}")

        # Assign the calculated value to the heatmap
        heatmap[y_bin, x_bin] = value

    return heatmap

def apply_blur(heatmap, blur, length_zones, width_zones):
    """
    Apply Gaussian blur to the heatmap for smoothing.

    Parameters:
        heatmap (np.ndarray): 2D heatmap array.
        blur (float or None): Standard deviation for Gaussian blur. Defaults to None.
        length_zones (array): Zone edges along pitch length.
        width_zones (array): Zone edges along pitch width.

    Returns:
        np.ndarray: Blurred heatmap.

    Example:
        blurred = apply_blur(heatmap, blur=5, length_zones, width_zones)
    """
    # calculate default blur if none is provided
    if blur is None:
        blur = max(1, min(len(length_zones), len(width_zones)) / 10)
    
    # apply Gaussian blur or return the original heatmap
    return gaussian_filter(heatmap, sigma=blur) if blur > 0 else heatmap


def get_colormap(blurred_heatmap, pitch_color, colors, color_range, stepped=False):
    """
    Generate a colormap for the heatmap based on defined range and colors.

    Parameters:
        blurred_heatmap (np.ndarray): Heatmap after applying Gaussian blur.
        pitch_color (str): Color for the pitch background.
        colors (str or list): Gradient colors for the colormap.
        color_range (tuple): Min and max values or percentiles for the colormap.
        stepped (bool): Use stepped color bins instead of smooth gradient.

    Returns:
        tuple: (colormap, normalization) - Matplotlib colormap and normalization.

    Example:
        cmap, norm = get_colormap(blurred_heatmap, pitch_color, colors, color_range)
    """
    # determine the minimum and maximum values for the colormap
    vmin, vmax = color_range

    # if vmin or vmax are given as percentiles, compute them from the heatmap values
    if isinstance(vmin, str) and vmin.endswith('%'):
        vmin = np.percentile(blurred_heatmap[blurred_heatmap > 0], float(vmin.strip('%')))
    if isinstance(vmax, str) and vmax.endswith('%'):
        vmax = np.percentile(blurred_heatmap[blurred_heatmap > 0], float(vmax.strip('%')))

    if stepped and isinstance(colors,list):
        # build N discrete bins for our exact hex list
        boundaries = np.linspace(vmin, vmax, len(colors) + 1)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)
    else:
        # smooth gradient
        palette = [pitch_color, colors] if isinstance(colors,str) else colors
        cmap = LinearSegmentedColormap.from_list("custom_heatmap", palette)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    return cmap, norm

def draw_heatmap(
    ax: plt.Axes,                   # Matplotlib axis to draw the heatmap on
    blurred_heatmap_array: np.ndarray,  # Blurred heatmap array for color intensity
    blurred_heatmap_size_array: np.ndarray,  # Blurred heatmap array for rectangle sizes
    length_zones: np.ndarray,       # Zone edges along pitch length
    width_zones: np.ndarray,        # Zone edges along pitch width
    cmap: plt.cm,                   # Colormap for the heatmap
    norm: plt.Normalize,            # Normalization for the heatmap values
    variable_size: bool = False,    # Enable dynamic sizing based on size array
    rounding: float = 0.8,          # Size of the rounded corners (relative to rectangle size)
    gutter: float = 0.0,            # Size of the gutter between rectangles (relative to zone size)
):
    """
    Draw the heatmap on the given axis with rounded corners, gutters, and optional dynamic sizing.
    When variable_size=True, the rounding decreases for smaller zones.

    Parameters:
        ax (plt.Axes): Matplotlib axis to draw the heatmap on.
        blurred_heatmap_array (np.ndarray): Blurred heatmap array for color intensity.
        blurred_heatmap_size_array (np.ndarray): Blurred heatmap array for rectangle sizes.
        length_zones (np.ndarray): Zone edges along pitch length.
        width_zones (np.ndarray): Zone edges along pitch width.
        cmap (plt.cm): Colormap for the heatmap.
        norm (plt.Normalize): Normalization for the heatmap values.
        variable_size (bool): If True, size rectangles based on blurred_heatmap_size_array.
        rounding (float): Base size of the rounded corners (relative to rectangle size).
        gutter (float): Size of the gutter between rectangles (relative to zone size).

    Returns:
        plt.Axes: Axis with the drawn heatmap.
    """
    # Initialize lists for rectangles and their corresponding colors
    rounded_rectangles = []
    colors_list = []

    # Get maximum value in the size array for scaling (used if variable_size=True)
    size_max = blurred_heatmap_size_array.max() if variable_size else 1

    # Iterate over all bins in the heatmap
    for x_bin in range(blurred_heatmap_array.shape[1]):  # Iterate over x-axis bins (columns)
        for y_bin in range(blurred_heatmap_array.shape[0]):  # Iterate over y-axis bins (rows)
            # Get the heatmap intensity value for the current bin
            value = blurred_heatmap_array[y_bin, x_bin] or blurred_heatmap_array.min()

            # Calculate size scaling factor for the current bin
            size_factor = (blurred_heatmap_size_array[y_bin, x_bin] / size_max) if variable_size and size_max > 0 else 1

            # Dynamically adjust rounding based on size factor (smaller zones have less rounding)
            adjusted_rounding = rounding * size_factor

            # Determine the edges of the rectangle for the current bin
            x_left, x_right = length_zones[x_bin], length_zones[x_bin + 1]
            y_bottom, y_top = width_zones[y_bin], width_zones[y_bin + 1]

            # Calculate width and height of the rectangle
            width = (x_right - x_left) * size_factor
            height = (y_top - y_bottom) * size_factor

            # Center the rectangle within its zone when size_factor < 1
            x_center = (x_left + x_right) / 2
            y_center = (y_bottom + y_top) / 2
            x_left = x_center - width / 2
            y_bottom = y_center - height / 2


            # Apply gutter adjustments
            if gutter > 0:
                # Calculate the absolute zone width and height
                zone_width = x_right - x_left
                zone_height = y_top - y_bottom

                # Calculate gutter sizes for x and y axes
                gutter_x = gutter * zone_width  # Gutter as fraction of zone width
                gutter_y = gutter * zone_height  # Gutter as fraction of zone height

                # Adjust the dimensions and positions
                x_left += gutter_x / 2
                y_bottom += gutter_y / 2
                width = max(0, zone_width - gutter_x)  # Ensure width doesn't go negative
                height = max(0, zone_height - gutter_y)  # Ensure height doesn't go negative

            # Create a FancyBboxPatch rectangle with adjusted rounding
            rect = FancyBboxPatch(
                (x_left, y_bottom),
                width,
                height,
                boxstyle=f"round,pad=0.0,rounding_size={adjusted_rounding}",  # Remove padding for precise alignment
                linewidth=0,  # No additional border width
            )
            rounded_rectangles.append(rect)

            # Get the color for the rectangle from the colormap and normalization
            color = cmap(norm(value))
            colors_list.append(color)

    # Create a PatchCollection for efficient rendering of all rectangles
    collection = PatchCollection(rounded_rectangles, edgecolor=colors_list)  # Match edgecolor to facecolor
    collection.set_facecolor(colors_list)

    # Add the PatchCollection to the axis
    ax.add_collection(collection)
    return ax


def draw_labels(
    ax: plt.Axes,
    heatmap_array: np.ndarray,
    length_zones: np.ndarray,
    width_zones: np.ndarray,
    aggregation: str,
    label_size: int,
    label_color: str,
    label_digits: int | None,
    label_threshold: float,
    dynamic_label_size: bool,
    stroke_color: str = futicolor.dark,  # Stroke color for the label
    stroke_width: float = 2,           # Stroke width
):
    """
    Draw labels on a heatmap to show values in each zone with a stroke outline.

    Parameters:
        ax (plt.Axes): Matplotlib axis to draw on.
        heatmap_array (np.ndarray): Array of heatmap values to label.
        length_zones (np.ndarray): Zone edges along the x-axis.
        width_zones (np.ndarray): Zone edges along the y-axis.
        aggregation (str): Type of aggregation used (e.g., "count", "proportion").
        label_size (int): Base size of the labels.
        label_color (str): Hex color for the label text (default is white).
        label_digits (int or None): Decimal places to round values to.
        label_threshold (float): Minimum value to display a label in a zone.
        dynamic_label_size (bool): Scale label size dynamically based on heatmap intensity.
        stroke_color (str): Hex color for the label stroke (default is futicolor.dark).
        stroke_width (float): Width of the stroke outline.
    """
    for x_bin in range(heatmap_array.shape[1]):
        for y_bin in range(heatmap_array.shape[0]):
            value = heatmap_array[y_bin, x_bin]

            # Skip labels for values below the threshold
            if value < label_threshold:
                continue

            # Determine the number of digits to round based on aggregation
            if label_digits is None:
                label_digits = 0 if aggregation == "proportion" else 1

            # Format the value
            if aggregation == "proportion":
                formatted_value = f"{int(round(value * 100))}%" if label_digits == 0 else f"{round(value * 100, label_digits)}%"
            else:
                formatted_value = f"{int(round(value))}" if label_digits == 0 else f"{round(value, label_digits)}"

            # Get zone center for placing text
            x_center = (length_zones[x_bin] + length_zones[x_bin + 1]) / 2
            y_center = (width_zones[y_bin] + width_zones[y_bin + 1]) / 2

            # Adjust label size dynamically based on value
            adjusted_label_size = label_size
            if dynamic_label_size:
                max_intensity = heatmap_array.max()
                intensity_factor = value / max_intensity if max_intensity > 0 else 1
                adjusted_label_size = max(8, label_size * intensity_factor)

            # Draw the label with stroke effect
            text = ax.text(
                x_center,
                y_center,
                formatted_value,
                color=label_color,  # Text color (e.g., white)
                fontsize=adjusted_label_size,
                ha="center",
                va="center",
                zorder=5,  # Ensure text appears above the heatmap
                family="Inter",
                path_effects=[
                    withStroke(linewidth=stroke_width, foreground=stroke_color)  # Stroke effect
                ],
            )

def plot_heatmap(
    pitch: PlotPitch,                # Soccer pitch instance
    fig: plt.Figure,                 # Matplotlib figure
    ax: plt.Axes,                    # Matplotlib axis
    data: pd.DataFrame,              # Event data
    aggregation: str = "count",      # Aggregation function for intensity
    value_column: str = None,        # Column for intensity aggregation
    x_zones: int | list | None = None,  # Number or positions of x-zones
    y_zones: int | list | None = None,  # Number or positions of y-zones
    blur: float | None = None,       # Gaussian blur standard deviation
    colors: str | list = futicolor.green,  # Gradient colors
    color_range: tuple = ("5%", "95%"),  # Range for colormap normalization
    stepped: bool = False,           # True for stepped color bins, false for smooth gradient
    rounding: float = 0.8,           # Rounded corner size
    gutter: float = 0.12,            # Gutter size
    variable_size: bool = False,     # Enable dynamic sizing
    show_labels: bool = False,       # Whether to display labels in zones
    label_size: int = 10,            # Font size for labels
    label_color: str = "#FFFFFF",    # Font color for labels
    label_digits: int | None = None, # Digits to round values to
    label_threshold: float = 0.1,    # Minimum value to display a label
    dynamic_label_size: bool = True, # Dynamically adjust label size based on intensity
    stroke_color: str = futicolor.dark,  # Stroke color for the label
    stroke_width: float = 2,           # Stroke width
):
    """
    Plot a heatmap on a soccer pitch with optional zone labels.

    Parameters:
        ... (same as before)
    """
    # Calculate pitch zones based on input parameters
    length_zones, width_zones = get_zones(pitch, x_zones, y_zones)

    # Transform data coordinates to align with the pitch orientation.
    flip = pitch.orientation == "tall"
    transformed_data = transform_xy(data=data, pitch=pitch, flip_coords=flip)

    # Generate intensity and size heatmaps
    heatmap_array = get_heatmap(transformed_data, length_zones, width_zones, aggregation, value_column)
    heatmap_size_array = get_heatmap(transformed_data, length_zones, width_zones, aggregation="proportion", value_column=None)

    # Apply Gaussian blur to both heatmaps
    blurred_heatmap_array = apply_blur(heatmap_array, blur, length_zones, width_zones)
    blurred_heatmap_size_array = apply_blur(heatmap_size_array, blur, length_zones, width_zones)

    # Generate colormap and normalization based on intensity values
    cmap, norm = get_colormap(blurred_heatmap_array, pitch.pitch_color, colors, color_range, stepped=stepped)

    # Draw the heatmap with dynamic or fixed sizing
    ax = draw_heatmap(
        ax=ax,
        blurred_heatmap_array=blurred_heatmap_array,
        blurred_heatmap_size_array=blurred_heatmap_size_array,
        length_zones=length_zones,
        width_zones=width_zones,
        cmap=cmap,
        norm=norm,
        variable_size=variable_size,
        rounding=rounding,
        gutter=gutter,
    )

    # Add labels if enabled
    if show_labels:
        draw_labels(
            ax=ax,
            heatmap_array=blurred_heatmap_array,
            length_zones=length_zones,
            width_zones=width_zones,
            aggregation=aggregation,
            label_size=label_size,
            label_color=label_color,
            label_digits=label_digits,
            label_threshold=label_threshold,
            dynamic_label_size=dynamic_label_size,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

    # Overlay pitch lines with transparency
    _overlay_pitch_markings(ax, pitch)

    return fig, ax

def plot_territory(
    pitch: PlotPitch,               # Soccer pitch instance defining pitch dimensions
    fig: plt.Figure,                # Matplotlib figure to plot on
    ax: plt.Axes,                   # Matplotlib axis to plot on
    data: pd.DataFrame,             # Event data, must include a 'home' column for team association
    x_zones: int | list | None = 21, # Number of x-zones or specific breakpoints along pitch length
    y_zones: int | list | None = 14, # Number of y-zones or specific breakpoints along pitch width
    blur: float | None = None,      # Gaussian blur standard deviation (smoothens heatmap values)
    rounding: float = 0.5,          # Size of the rounded corners for rectangles (relative to zone size)
    gutter: float = 0.1,            # Size of gaps between rectangles (relative to zone size)
    variable_size: bool = True,     # Scale rectangles dynamically based on zone intensity
    show_labels: bool = False,      # Display numerical labels within zones
    label_size: int = 10,           # Base font size for numerical labels
    label_color: str = "#FFFFFF",   # Color of the numerical labels
    label_digits: int | None = None, # Number of decimal places for numerical labels (default varies)
    label_threshold: float = 0.1,   # Minimum zone intensity required to display a label
    dynamic_label_size: bool = True, # Scale label size dynamically based on intensity
    boundaries: list = [0, 0.5, 1], # List of boundary values for colormap segmentation
    colors: list = None,            # List of colors corresponding to boundaries
    stroke_color: str = futicolor.dark,  # Stroke color for the label
    stroke_width: float = 2,           # Stroke width
    pitch_line_color: str = futicolor.light,  # Color for pitch markings and goal lines
):
    """
    Plot a territory heatmap on a soccer pitch, showing the proportion of home touches in each zone.

    Parameters:
        pitch (PlotPitch): Soccer pitch instance defining pitch properties.
        fig (plt.Figure): Matplotlib figure to plot on.
        ax (plt.Axes): Matplotlib axis to plot on.
        data (pd.DataFrame): Event data containing a 'home' column indicating whether an event belongs to the home team.
        x_zones (int | list | None): Number of x-zones or custom breakpoints along the pitch length. Defaults to 21 zones.
        y_zones (int | list | None): Number of y-zones or custom breakpoints along the pitch width. Defaults to 14 zones.
        blur (float | None): Standard deviation for Gaussian blur to smooth the heatmap. Defaults to None (no blur).
        rounding (float): Rounded corner size for rectangles (relative to the zone size). Default is 0.5.
        gutter (float): Gap size between rectangles (relative to the zone size). Default is 0.1.
        variable_size (bool): Whether to scale rectangle sizes dynamically based on zone intensity. Defaults to True.
        show_labels (bool): If True, numerical labels are displayed in each zone. Default is False.
        label_size (int): Base font size for numerical labels. Default is 10.
        label_color (str): Font color for numerical labels. Default is white ("#FFFFFF").
        label_digits (int | None): Number of decimal places for numerical labels. Default varies by context.
        label_threshold (float): Minimum value required to display a label in a zone. Default is 0.1.
        dynamic_label_size (bool): If True, scales label size dynamically based on zone intensity. Default is True.
        boundaries (list): Boundary values for colormap segmentation. Default is [0, 0.5, 1].
        colors (list): Colors corresponding to boundaries. Default is None (uses blue-to-pink gradient).

    Returns:
        tuple: Updated figure and axis with the territory heatmap plotted.
    """
    # Step 1: Define pitch zones
    length_zones, width_zones = get_zones(pitch, x_zones, y_zones)

    # Step 2: Separate data into home and away teams based on the 'home' column
    home_df = data[data["home"] == "home"]  # Home team data
    away_df = data[data["home"] == "away"]  # Away team data

    # Step 3: Transform coordinates for home and away teams
    home_df = transform_xy(data=home_df, pitch=pitch, flip_coords=False)  # Home team events stay unflipped
    away_df = transform_xy(data=away_df, pitch=pitch, flip_coords=True)   # Away team events flipped for alignment

    # Step 4: Generate heatmaps for home and away teams
    home_heatmap = get_heatmap(
        transformed_data=home_df,
        length_zones=length_zones,
        width_zones=width_zones,
        aggregation="count",
        value_column=None,
    )
    away_heatmap = get_heatmap(
        transformed_data=away_df,
        length_zones=length_zones,
        width_zones=width_zones,
        aggregation="count",
        value_column=None,
    )

    # Step 5: Combine home and away heatmaps to calculate total touches per zone
    total_touches = home_heatmap + away_heatmap

    # Step 6: Calculate the proportion of home touches in each zone
    with np.errstate(divide="ignore", invalid="ignore"):
        territory_heatmap = np.divide(home_heatmap, total_touches, where=total_touches > 0)
        territory_heatmap[total_touches == 0] = 0.5  # Assign neutral value for empty zones

    # Step 7: Apply Gaussian blur for smoothing, if specified
    blurred_heatmap_array = apply_blur(territory_heatmap, blur, length_zones, width_zones)
    blurred_heatmap_size_array = apply_blur(total_touches, blur, length_zones, width_zones)

    # Step 8: Create colormap and normalization
    if colors is None:
        colors = ["#FF69B4", "#FFC0CB", "#ADD8E6", "#0000FF"]  # Default pink-to-blue gradient
    cmap = ListedColormap(colors)  # Create colormap from provided colors
    norm = BoundaryNorm(boundaries, len(colors))  # Normalize values to match boundaries

    # Step 9: Draw heatmap on the pitch
    ax = draw_heatmap(
        ax=ax,
        blurred_heatmap_array=blurred_heatmap_array,
        blurred_heatmap_size_array=blurred_heatmap_size_array,
        length_zones=length_zones,
        width_zones=width_zones,
        cmap=cmap,
        norm=norm,
        variable_size=variable_size,
        rounding=rounding,
        gutter=gutter,
    )

    # Step 10: Add labels if enabled
    if show_labels:
        draw_labels(
            ax=ax,
            heatmap_array=blurred_heatmap_array,
            length_zones=length_zones,
            width_zones=width_zones,
            aggregation="proportion",
            label_size=label_size,
            label_color=label_color,
            label_digits=label_digits,
            label_threshold=label_threshold,
            dynamic_label_size=dynamic_label_size,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

    # Step 11: Overlay pitch lines for clarity
    _overlay_pitch_markings(ax, pitch)

    return fig, ax


def plot_zscore_heatmap(
    pitch: PlotPitch,
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    group_col: Optional[str] = None,
    group_val: Optional[Union[str, int]] = None,
    mean_heatmap: Optional[np.ndarray] = None,
    std_heatmap: Optional[np.ndarray] = None,
    x_zones: Union[int, List[float], None] = None,
    y_zones: Union[int, List[float], None] = None,
    blur: Optional[float] = None,
    colors: Union[str, List[str]] = "plasma",
    color_range: Tuple[float, float] = (-2, 2),
    stepped: bool = False,
    rounding: float = 0,
    gutter: float = 0,
    flip_coords: bool = False,
    **draw_kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a z-score heatmap for a specific group (e.g. a role) or use provided global statistics.

    If mean_heatmap and std_heatmap are both None, computes them across all unique values in group_col.
    Then computes the per-zone counts for group_val, converts to z-scores, and draws the heatmap.

    Parameters:
        pitch: PlotPitch instance defining pitch layout.
        fig, ax: Matplotlib figure and axis to draw into.
        data: DataFrame with event coordinates and grouping column.
        group_col: Column name to group by (e.g. 'role') when computing global stats.
        group_val: Single group value to plot (e.g. one role name).
        mean_heatmap: Precomputed global mean per-zone array (optional).
        std_heatmap: Precomputed global std per-zone array (optional).
        x_zones, y_zones: Zone definitions (count or break list).
        blur: Sigma for Gaussian blur.
        colors: Name or list for colormap.
        color_range: (min, max) for colormap normalization.
        stepped: True for stepped color bins, false for smooth gradient.
        rounding, gutter: Style params for heatmap cells.
        flip_coords: Whether to flip coordinates for alignment.
        draw_kwargs: Additional kwargs passed through to draw_heatmap.

    Returns:
        fig, ax with the plotted z-score heatmap for the specified group_val.
    """
    # define pitch zones
    pitch.construct_pitch()
    length_zones, width_zones = get_zones(pitch, x_zones, y_zones)

    # transform coordinates
    transformed = transform_xy(data=data, pitch=pitch, flip_coords=flip_coords)

    # compute global mean/std if not provided
    if mean_heatmap is None or std_heatmap is None:
        if group_col is None:
            raise ValueError("group_col must be provided when mean/std not passed in")
        groups = transformed[group_col].dropna().unique()
        raw_counts = []
        for grp in groups:
            df_grp = transformed[transformed[group_col] == grp]
            if df_grp.empty:
                continue
            counts = get_heatmap(
                transformed_data=df_grp,
                length_zones=length_zones,
                width_zones=width_zones,
                aggregation='count',
                value_column=None
            )
            raw_counts.append(counts)
        if not raw_counts:
            raise ValueError(f"No data found for any groups in {group_col}")
        stack = np.stack(raw_counts, axis=0)
        mean_heatmap = stack.mean(axis=0)
        std_heatmap = stack.std(axis=0)

    # ensure group_val for plotting
    if group_val is None:
        raise ValueError("group_val must be provided to select which group's z-score to plot")

    # compute counts for the specified group
    subset = transformed[transformed[group_col] == group_val] if group_col else transformed
    if subset.empty:
        raise ValueError(f"No data for group '{group_val}' in column {group_col}")
    counts_hm = get_heatmap(
        transformed_data=subset,
        length_zones=length_zones,
        width_zones=width_zones,
        aggregation='count',
        value_column=None
    )

    # compute z-scores and blur
    zscore_hm = (counts_hm - mean_heatmap) / std_heatmap
    blurred = apply_blur(zscore_hm, blur, length_zones, width_zones)

    # build colormap and normalization
    cmap, norm = get_colormap(
        blurred,
        pitch_color=pitch.pitch_color,
        colors=colors,
        color_range=color_range,
        stepped=stepped
    )

    # draw heatmap cells
    draw_heatmap(
        ax=ax,
        blurred_heatmap_array=blurred,
        blurred_heatmap_size_array=blurred,
        length_zones=length_zones,
        width_zones=width_zones,
        cmap=cmap,
        norm=norm,
        rounding=rounding,
        gutter=gutter,
        **draw_kwargs
    )

    # overlay transparent pitch lines
    _overlay_pitch_markings(ax, pitch)

    return fig, ax