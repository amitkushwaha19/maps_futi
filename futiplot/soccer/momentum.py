import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define default colors globally for consistent styling
DEFAULT_COLORS = {
    "home": "#00B7FF",  # Light blue for Home
    "away": "#EA1F96",  # Pink for Away
    "difference": "#ffffff",  # White for difference bars
}

def create_time_windows(data, time_column, time_window):
    """
    Create time windows for the specified time column.

    Parameters:
        data (pd.DataFrame): Input data containing a time column.
        time_column (str): The column representing time in seconds.
        time_window (int): The size of each time window in seconds.

    Returns:
        pd.DataFrame: Data with an added 'time_window' column.
    """
    data["time_window"] = (data[time_column] // time_window) * time_window
    return data

def summarize_by_window(data, time_window_column, home_column, value_column, agg_function):
    """
    Summarize data by time window and home/away status using the specified aggregation function.

    Parameters:
        data (pd.DataFrame): Input data containing home and time window columns.
        time_window_column (str): The column representing time windows.
        home_column (str): The column indicating home/away status (boolean).
        value_column (str): The column to aggregate.
        agg_function (callable): The aggregation function (e.g., sum, mean).

    Returns:
        pd.DataFrame: Pivot table summarizing momentum data by time window and home/away status.
    """
    summary = data.groupby([home_column, time_window_column])[value_column].agg(agg_function).reset_index()
    pivot = summary.pivot(index=time_window_column, columns=home_column, values=value_column).fillna(0)
    pivot.columns = ["Away", "Home"]  # Rename columns for clarity
    return pivot

def apply_rolling_window(momentum_data, rolling_window, weighted=False):
    """
    Apply a rolling window to the momentum data for smoothing.

    Parameters:
        momentum_data (pd.DataFrame): Aggregated momentum data with 'Home' and 'Away' columns.
        rolling_window (int): Size of the rolling window.
        weighted (bool): Whether to use Linear Weighted Moving Average (LWMA).

    Returns:
        pd.DataFrame: Smoothed momentum data.
    """
    if rolling_window > 1:
        rolling = momentum_data.rolling(window=rolling_window, min_periods=1, center=False)
        if weighted:
            return rolling.apply(
                lambda series: np.dot(series, np.arange(1, len(series) + 1)) / np.arange(1, len(series) + 1).sum(),
                raw=True,
            )
        else:
            return rolling.mean()
    return momentum_data

def draw_momentum(momentum_data, time_window, colors=None, gap=0, ax=None):
    """
    Draw a minimal momentum chart with home and away bars.

    Parameters:
        momentum_data (pd.DataFrame): Aggregated momentum data with 'Home' and 'Away' columns.
        time_window (int): The size of each time window in seconds.
        colors (dict): Dictionary specifying colors for 'home', 'away', and 'difference' bars.
        gap (float): Vertical gap between home and away bars (default: 0).
        ax (matplotlib.axes.Axes): Pre-existing axis to draw the plot on.

    Returns:
        matplotlib.axes.Axes: The axis with the momentum plot.
    """
    bar_width = time_window * 0.8  # Bars take up 80% of the time window width
    if colors is None:
        colors = DEFAULT_COLORS

    # Plot Home and Away bars with the gap applied only during plotting
    home = momentum_data["Home"]
    away = momentum_data["Away"]
    ax.bar(momentum_data.index, home, bottom=gap, width=bar_width, color=colors["home"], alpha=0.4, zorder=2)
    ax.bar(momentum_data.index, -away, bottom=-gap, width=bar_width, color=colors["away"], alpha=0.4, zorder=2)

    # Plot the difference bars
    for time, diff in zip(momentum_data.index, home - away):
        diff_gap = gap if diff > 0 else -gap
        fill_color = colors["home"] if diff > 0 else colors["away"]
        ax.bar(time, diff, bottom=diff_gap, width=bar_width, color=fill_color, alpha=1, zorder=3)

    # Adjust the y-axis to include the gap
    ax.set_ylim(-15 - gap, 15 + gap)
    ax.axis("off")
    return ax

def process_half(half_data, time_column, home_column, value_column, agg_function, time_window, rolling_window):
    """
    Process data for a single half.

    Parameters:
        half_data (pd.DataFrame): Input data for the specific half.
        time_column (str): The column representing time in seconds.
        home_column (str): The column indicating home/away status.
        value_column (str): The column to aggregate.
        agg_function (callable): Aggregation function for summarizing data.
        time_window (int): The size of each time window in seconds.
        rolling_window (int): Size of the rolling window for smoothing.

    Returns:
        pd.DataFrame: Processed momentum data for the half.
    """
    half_data = create_time_windows(half_data, time_column, time_window)
    momentum_data = summarize_by_window(
        half_data,
        time_window_column="time_window",
        home_column=home_column,
        value_column=value_column,
        agg_function=agg_function,
    )
    all_time_windows = pd.RangeIndex(
        start=momentum_data.index.min(),
        stop=momentum_data.index.max() + time_window,
        step=time_window,
    )
    momentum_data = momentum_data.reindex(all_time_windows, fill_value=0)
    momentum_data = apply_rolling_window(momentum_data, rolling_window)
    return momentum_data

def plot_momentum(
    data,
    figsize=(15, 5),  # Overall figure size
    value_column=None,
    filter="x_start >= 105 * 2 / 3",
    time_column="time_seconds",
    home_column="home",
    time_window=120,
    rolling_window=2,
    function="count",
    colors=None,
):
    """
    Calculate and plot momentum with rolling statistics for each half.

    Parameters:
        data (pd.DataFrame): The input data containing time, value, home columns, and period_id.
        figsize (tuple): Size of the overall figure (default: (15, 5)).
        value_column (str or None): The column representing the values to summarize. If None, counts rows matching the filter.
        filter (str): A string-based filter condition applied to the DataFrame (default: "x_start >= 105 * 2 / 3").
        time_column (str): The column representing time in seconds (default: "time_seconds").
        home_column (str): The column indicating whether the team is home (True for home, False for away).
        time_window (int): The size of each time window in seconds (default: 120).
        rolling_window (int): The number of time windows to use for the rolling calculation (default: 2).
        function (str or callable): The aggregation function to apply ("count", "mean", "sum", or callable).
        colors (dict): Optional dictionary specifying colors for teams and difference bars.

    Returns:
        tuple: Matplotlib figure and axes.
    """
    # Validate the aggregation function
    if isinstance(function, str):
        agg_function = {"count": pd.Series.count, "mean": pd.Series.mean, "sum": pd.Series.sum}.get(function)
        if agg_function is None:
            raise ValueError("Unsupported function. Use 'count', 'mean', or 'sum'.")
    elif callable(function):
        agg_function = function
    else:
        raise TypeError("Function must be a string ('count', 'mean', 'sum') or a callable.")

    data = data.query(filter).copy()
    temp_column = "_temp_count"
    if value_column is None:
        data[temp_column] = 1
        value_column = temp_column

    required_columns = {value_column, time_column, home_column, "period_id"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Data must contain the columns: {required_columns}")

    halves = {1: data[data["period_id"] == 1], 2: data[data["period_id"] == 2]}
    half_durations = {p: h[time_column].max() - h[time_column].min() for p, h in halves.items() if not h.empty}
    total_duration = sum(half_durations.values())
    fig_width, fig_height = figsize
    subplot_widths = {p: fig_width * (d / total_duration) for p, d in half_durations.items()}

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [subplot_widths[1], subplot_widths[2]], "wspace": 0},
    )

    for period, ax in zip(halves.keys(), axes):
        momentum_data = process_half(
            halves[period], time_column, home_column, value_column, agg_function, time_window, rolling_window
        )
        draw_momentum(momentum_data, time_window, colors, gap=0, ax=ax)

    if temp_column in data.columns:
        data.drop(columns=[temp_column], inplace=True)

    return fig, axes
