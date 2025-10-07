import pandas as pd
import matplotlib.pyplot as plt

# Define default colors globally for consistent styling
DEFAULT_COLORS = {
    "home": "#00B7FF",  # Light blue for Home
    "away": "#EA1F96",  # Pink for Away
}

def calculate_team_cumsum(data, time_column, xg_column, team_filter, period_start, period_end):
    """
    Calculate cumulative xG for a specific team (Home or Away) and ensure connections at both ends.

    Parameters:
        data (pd.DataFrame): Input data containing time and xG columns.
        time_column (str): The column representing time in seconds.
        xg_column (str): The column representing xG values.
        team_filter (pd.Series): Boolean mask for filtering team-specific rows.
        period_start (int): The starting time of the period in seconds.
        period_end (int): The ending time of the period in seconds.

    Returns:
        pd.DataFrame: DataFrame with time and cumulative xG for the specific team.
    """
    team_data = data[team_filter].copy()
    team_data = team_data.sort_values(by=[time_column])
    team_data["cumulative_xg"] = team_data[xg_column].cumsum()

    # Add points for the start and end of the period
    start_row = pd.DataFrame({
        time_column: [period_start],
        "cumulative_xg": [0]  # Start at xG = 0 or from previous period if applicable
    })
    end_row = pd.DataFrame({
        time_column: [period_end],
        "cumulative_xg": [team_data["cumulative_xg"].iloc[-1] if not team_data.empty else 0]  # Extend to period_end
    })
    team_data = pd.concat([start_row, team_data[[time_column, "cumulative_xg"]], end_row], ignore_index=True)

    return team_data

def plot_xg_step(data, figsize=(15, 5), time_column="time_seconds", 
                 home_column="home", xg_column="xg", colors=None):
    """
    Plot an xG Step Chart showing cumulative xG for Home and Away teams, with continuity across periods.

    Parameters:
        data (pd.DataFrame): Input data containing time, team (home/away), and xG columns.
        figsize (tuple): Size of the overall figure (default: (15, 5)).
        time_column (str): The column representing time in seconds (default: "time_seconds").
        home_column (str): The column indicating team status ("home" or "away").
        xg_column (str): The column representing xG values (default: "xg").
        colors (dict): Optional dictionary specifying colors for 'home' and 'away'.

    Returns:
        tuple: Matplotlib figure and axes.
    """
    if colors is None:
        colors = DEFAULT_COLORS

    # Split data by periods (e.g., halves or any other period_id defined)
    periods = sorted(data["period_id"].unique())
    fig, axes = plt.subplots(ncols=len(periods), figsize=figsize, gridspec_kw={"wspace": 0}, sharey=True)
    fig.patch.set_alpha(0)  # Transparent background

    # Initialize cumulative totals for continuity between periods
    cumulative_start = {"home": 0, "away": 0}

    for period, ax in zip(periods, axes):
        # Filter data for the current period
        period_data = data[data["period_id"] == period]

        # Determine the start and end times for this period
        period_start = period_data[time_column].min()
        period_end = period_data[time_column].max()

        # Calculate cumulative xG for Home and Away separately
        home_cumsum = calculate_team_cumsum(
            period_data, time_column, xg_column,
            team_filter=(period_data[home_column] == "home"),  # Filter for home team rows
            period_start=period_start, period_end=period_end
        )
        away_cumsum = calculate_team_cumsum(
            period_data, time_column, xg_column,
            team_filter=(period_data[home_column] == "away"),  # Filter for away team rows
            period_start=period_start, period_end=period_end
        )

        # Adjust cumulative totals for continuity (if this is not the first period)
        if period > 1:
            home_cumsum["cumulative_xg"] += cumulative_start["home"]
            away_cumsum["cumulative_xg"] += cumulative_start["away"]

        # Update cumulative totals for the next period
        if not home_cumsum.empty:
            cumulative_start["home"] = home_cumsum["cumulative_xg"].iloc[-1]
        if not away_cumsum.empty:
            cumulative_start["away"] = away_cumsum["cumulative_xg"].iloc[-1]

        # Plot Home xG
        ax.step(
            home_cumsum[time_column],
            home_cumsum["cumulative_xg"],
            where="post",
            color=colors["home"],
            lw=4,
        )

        # Plot Away xG
        ax.step(
            away_cumsum[time_column],
            away_cumsum["cumulative_xg"],
            where="post",
            color=colors["away"],
            lw=4,
        )

        # Minimal aesthetics: remove labels, axes, and ticks
        ax.axis("off")

    return fig, axes
