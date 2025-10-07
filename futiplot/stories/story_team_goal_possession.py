import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from matplotlib.axes import Axes
import math
from futiplot.format import create_story_axes, draw_event_legend
from futiplot.soccer.pitch import plot_pitch
from futiplot.soccer.events import plot_passage
from futiplot.utils.colors import futicolor
from futiplot.utils.fontutils import get_font
import pandas as pd
from sqlalchemy import create_engine, text
 
 
def _get_goal_info(df):
    # locate the goal event (assumes exactly one per df)
    goal = df[df["goal"] == 1].iloc[0]
 
    # extract scores and home/away flag
    gf = goal["goals_for"]
    ga = goal["goals_against"]
    is_home = (goal["home"] == "home")
 
    # compute rounded‑up minute
    seconds = goal["time_seconds"]
    period = goal["period_id"]
    minutes = math.ceil(seconds / 60)
 
    # format stoppage time if needed
    if period == 1 and minutes > 45:
        minute_str = f"45+{minutes - 45}'"
    elif period == 2 and minutes > 90:
        minute_str = f"90+{minutes - 90}'"
    else:
        minute_str = f"{minutes}'"
 
    # pull scorer name
    scorer_name = goal["player_name"]  # Using actual player name from live data
    is_home_scorer = is_home
 
    # decide display order and colors
    if is_home:
        left_text, right_text = str(gf), str(ga)
        left_color, right_color = futicolor.blue, futicolor.light
    else:
        left_text, right_text = str(ga), str(gf)
        left_color, right_color = futicolor.light, futicolor.pink
 
    return (
        left_text, right_text,
        left_color, right_color,
        minute_str, scorer_name,
        is_home_scorer
    )
 
 
def _draw_score_title(
    ax: Axes,
    left_text: str,
    right_text: str,
    left_color: str,
    right_color: str,
    minute_str,
    scorer_name,
    is_home_scorer,
    offset: float = 0.06,
    fontsize: int = 60
):
    """
    Render the score title on the provided axis.
 
    A light-colored hyphen is centered, with the two scores placed
    just to its left and right, using the specified colors.
    """
    ax.axis("off")
    ax.set_facecolor(futicolor.dark)
    font_prop = get_font("bold")
    font_prop.set_size(fontsize)
 
    ax.text(
        0.5, 0.5, "-",
        fontproperties=font_prop,
        color=futicolor.light,
        ha="center", va="center",
        transform=ax.transAxes
    )
    ax.text(
        0.5 - offset, 0.5, left_text,
        fontproperties=font_prop,
        color=left_color,
        ha="right", va="center",
        transform=ax.transAxes
    )
    ax.text(
        0.5 + offset, 0.5, right_text,
        fontproperties=font_prop,
        color=right_color,
        ha="left", va="center",
        transform=ax.transAxes
    )
 
    # draw match minute below
    minute_font = get_font("regular")
    minute_font.set_size(20)
    ax.text(0.5, 0.5 - 0.30, minute_str,
            fontproperties=minute_font,
            color=futicolor.light,
            ha="center", va="center",
            transform=ax.transAxes)
 
    # draw scorer name to left or right of minute
    name_font = get_font("regular")
    name_font.set_size(20)
    if is_home_scorer:
        x_pos, halign, name_color = 0.5 - offset, "right", futicolor.light
    else:
        x_pos, halign, name_color = 0.5 + offset, "left", futicolor.light
 
    ax.text(x_pos, 0.5 - 0.30, scorer_name,
            fontproperties=name_font,
            color=name_color,
            ha=halign, va="center",
            transform=ax.transAxes)
 
 
def story_team_goal_possession(
    df,
    *,
    fig_kwargs=None,
    gridspec_kw=None
):
    """
    Assemble a story-style figure of a goal possession sequence.
 
    The layout consists of a title row showing the current score,
    the pitch with the possession events, and a legend explaining symbols.
    """
    mosaic = [["title"], ["pitch"], ["legend"]]
    fig, axes = create_story_axes(
        mosaic,
        fig_kwargs=fig_kwargs or {
            "figsize": (10.8, 17.8),
            "dpi": 100,
            "facecolor": futicolor.dark
        },
        gridspec_kw=gridspec_kw or {
            "width_ratios": [1],
            "height_ratios": [2, 10, 0.5],
            "wspace": 0.0,
            "hspace": 0.0
        },
        constrained_layout=False
    )
 
    # get goal info
    (
        left_text, right_text,
        left_color, right_color,
        minute_str, scorer_name,
        is_home_scorer
    ) = _get_goal_info(df)
 
    # draw title
    _draw_score_title(
        axes["title"],
        left_text, right_text,
        left_color, right_color,
        minute_str, scorer_name, is_home_scorer
    )
 
    ax_pitch = axes["pitch"]
    _, ax_pitch, pitch = plot_pitch(
        ax=ax_pitch,
        line_color=futicolor.light1,
        linewidth=2,
        show_legend=False
    )
   
    plot_passage(df, ax=ax_pitch, pitch=pitch)
 
    ax_legend = axes["legend"]
    ax_legend.axis("off")
    ax_legend.set_facecolor(futicolor.dark)
    draw_event_legend(
        ax_legend,
        orientation="horizontal",
        symbol_color=futicolor.light,
        label_color=futicolor.light1,
        label_fontsize=16,
        padding=0.1
    )
   
    return fig, axes
 
 
if __name__ == "__main__":
    # Database connection parameters
    db_params = {
        "dbname": "futi",
        "user": "postgres",
        "password": "root",
        "host": "localhost",
        "port": "5432"
    }
 
    try:
        # Create SQLAlchemy engine
        connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        engine = create_engine(connection_string)
 
        # Query to fetch game IDs from storiesGameIds table
        match_id_query = """
        SELECT match_id
        FROM stories_game_ids
        """
        match_ids_df = pd.read_sql(match_id_query, engine)
       
        if match_ids_df.empty:
            print("No game IDs found in storiesGameIds table. Exiting.")
            sys.exit(0)
 
        # Process each match_id
        for match_id in match_ids_df['match_id']:
            print(f"Processing match_id: {match_id}")
            game_dir = "story_images/game/"+str(match_id)
            os.makedirs(game_dir, exist_ok=True)
 
            # Query to fetch match data for the current match_id
            match_query = """
            SELECT V.*, p.player_name, p.player_nickname, p.player_lastname
            FROM vaep_mls AS V
            INNER JOIN players_mls AS p ON p.player_id = V.player_id
            WHERE match_id = %s
            """
           
            # Load data into DataFrame using SQLAlchemy
            df = pd.read_sql(match_query, engine, params=(match_id,))
            print(f"Loaded {len(df)} rows from the database for match_id {match_id}.")
 
            # Check if there are any goals
            if df.empty:
                print(f"No goals found for match_id {match_id}. Skipping.")
                continue
 
            # Generate the figure for each possession with a goal
            for possession_id in df['possession_id'].unique():
                print(f"Processing possession_id: {possession_id} for match_id: {match_id}")
                # Filter data for specific possession
                tempDF = df[df['possession_id'] == possession_id]
               
                # Skip if no goal in this possession
                if tempDF[(tempDF['possession_id'] == possession_id) & (tempDF['goal'] == 1)].empty:
                    continue
                   
                fig, _ = story_team_goal_possession(tempDF)
 
                # Save the figure to a PNG file with match_id included in filename
                output_path = os.path.join(game_dir, f"{match_id}_{possession_id}.png")
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
            # Delete the match_id from storiesGameIds after successful processing
            delete_query = """
            DELETE FROM stories_game_ids WHERE match_id = :match_id
            """
            with engine.connect() as conn:
                conn.execute(text(delete_query), {"match_id": match_id})
                conn.commit()
            print(f"Deleted match_id {match_id} from stories_game_ids table.")
 
    except Exception as e:
        print(f"Error connecting to database or processing data: {e}")
       
    finally:
        # Dispose of the engine
        if 'engine' in locals():
            engine.dispose()
 