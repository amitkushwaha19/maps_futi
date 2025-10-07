import pandas as pd
import numpy as np
import re
from shapely.geometry import Point, Polygon, LineString
from shapely.vectorized import contains

#globals for dribble derivation parameters
min_dribble_length = 3.0
max_dribble_length = 75.0
max_dribble_duration = 12.0

# keys to match action, result, and bodypart IDs to action and result names ----
spadl_action_df = pd.DataFrame({
    'type_name': [
        "pass",
        "cross",
        "throw_in",
        "freekick_crossed",
        "freekick_short",
        "corner_crossed",
        "corner_short",
        "take_on",
        "foul",
        "tackle",
        "interception",
        "shot",
        "shot_penalty",
        "shot_freekick",
        "keeper_save",
        "keeper_claim",
        "keeper_punch",
        "keeper_pick_up",
        "clearance",
        "bad_touch",
        "non_action",
        "dribble",
        "goalkick",
        "through_ball"
    ],
    'type_id': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ]
})

spadl_result_df = pd.DataFrame({
    'result_name': [
        "fail",
        "success",
        "offside",
        "owngoal",
        "yellow_card",
        "red_card",
        "out",
        "shot"
    ],
    'result_id': [
        0, 1, 2, 3, 4, 5, 6, 7
    ]
})

spadl_bodypart_df = pd.DataFrame({
    'bodypart_name': [
        "foot",
        "head",
        "other",
        "head/other",
        "foot_left",
        "foot_right"
    ],
    'bodypart_id': [
        0, 1, 2, 3, 4, 5
    ]
})

spadl_phase_names = ['kickoff', 'attacking_throw_in', 'possession_throw_in', 'attacking_freekick', 'possession_freekick',
               'long_goalkick', 'short_goalkick', 'corner',
               'corner_second_phase','attacking_freekick_second_phase',
               'accelerated_possession', 'buildup', 'progression', 'finishing',
               'securing_possession', 'high_transition', 'counterattack',
               'indirect_freekick','penalty','loose_ball','high_ball']

def mark_crosses_sb(df):
    # Define zones
    wide_flank_left = Polygon([(72, 0), (120, 0), (120, 18), (72, 18)])
    wide_flank_right = Polygon([(72, 62), (120, 62), (120, 80), (72, 80)])
    triangle_left = Polygon([(120, 30), (102, 18), (120, 18)])
    triangle_right = Polygon([(120, 50), (102, 62), (120, 62)])
    box_left = Polygon([(102, 50), (120, 50), (102, 18), (120, 18)])
    box_right = Polygon([(102, 30), (120, 30), (102, 62), (120, 62)])

    # Extract arrays
    x_start = df['x_start'].to_numpy(dtype=np.float64)
    y_start = df['y_start'].to_numpy(dtype=np.float64)
    x_end = df['x_end'].to_numpy(dtype=np.float64)
    y_end = df['y_end'].to_numpy(dtype=np.float64)
    name= df['name'].to_numpy()

    # Vectorized contains checks for start points
    in_flank_left = contains(wide_flank_left, x_start, y_start)
    in_flank_right = contains(wide_flank_right, x_start, y_start)
    in_triangle_left = contains(triangle_left, x_start, y_start)
    in_triangle_right = contains(triangle_right, x_start, y_start)

    #check for a pass
    is_pass = df['name'] == 'pass'

    # Determine side
    from_left = np.logical_and(is_pass, in_flank_left | in_triangle_left)
    from_right = np.logical_and(is_pass, in_flank_right | in_triangle_right)

    # Initialize result array
    is_cross = np.zeros(len(df), dtype=bool)

    # Build lines and test for intersections for left-origin passes
    if np.any(from_left):
        left_idx = np.where(from_left)[0]
        left_lines = [LineString([(x_start[i], y_start[i]), (x_end[i], y_end[i])]) for i in left_idx]
        left_hits = [line.intersects(box_right) for line in left_lines]
        is_cross[left_idx] = left_hits

    # Same for right-origin passes
    if np.any(from_right):
        right_idx = np.where(from_right)[0]
        right_lines = [LineString([(x_start[i], y_start[i]), (x_end[i], y_end[i])]) for i in right_idx]
        right_hits = [line.intersects(box_left) for line in right_lines]
        is_cross[right_idx] = right_hits

    # Assign result
    df['cross'] = is_cross
    return df

def mark_crosses_wyscout(df):
    # Define zones
    wide_flank_left = Polygon([(60, 0), (100, 0), (100, 22.5), (60, 22.5)])
    wide_flank_right = Polygon([(60, 77.5), (100, 77.5), (100, 100), (60, 100)])
    triangle_left = Polygon([(100, 37.5), (85, 22.5), (100, 22.5)])
    triangle_right = Polygon([(100, 50), (85, 77.5), (100, 77.5)])
    box_left = Polygon([(85, 50), (100, 50), (85, 22.5), (100, 22.5)])
    box_right = Polygon([(85, 37.5), (100, 37.5), (85, 77.5), (100, 77.5)])

    # Extract arraysAdd commentMore actions
    # Extract arrays
    x_start = df['x_start'].to_numpy(dtype=np.float64)
    y_start = df['y_start'].to_numpy(dtype=np.float64)
    x_end = df['x_end'].to_numpy(dtype=np.float64)
    y_end = df['y_end'].to_numpy(dtype=np.float64)
    name= df['type.primary'].to_numpy()

    # Vectorized contains checks for start points
    in_flank_left = contains(wide_flank_left, x_start, y_start)
    in_flank_right = contains(wide_flank_right, x_start, y_start)
    in_triangle_left = contains(triangle_left, x_start, y_start)
    in_triangle_right = contains(triangle_right, x_start, y_start)

    #check for a pass
    is_pass = (name == 'pass')

    # Determine side
    from_left = np.logical_and(is_pass, in_flank_left | in_triangle_left)
    from_right = np.logical_and(is_pass, in_flank_right | in_triangle_right)

    # Initialize result array
    is_cross = np.zeros(len(df), dtype=bool)

    # Build lines and test for intersections for left-origin passes
    if np.any(from_left):
        left_idx = np.where(from_left)[0]
        left_lines = [LineString([(x_start[i], y_start[i]), (x_end[i], y_end[i])]) for i in left_idx]
        left_hits = [line.intersects(box_right) for line in left_lines]
        is_cross[left_idx] = left_hits

    # Same for right-origin passes
    if np.any(from_right):
        right_idx = np.where(from_right)[0]
        right_lines = [LineString([(x_start[i], y_start[i]), (x_end[i], y_end[i])]) for i in right_idx]
        right_hits = [line.intersects(box_left) for line in right_lines]
        is_cross[right_idx] = right_hits

    # Assign result
    df['cross'] = is_cross
    return df

def order_tackles(df):
    # This is a correction for the fact that opta is not consistent in ordering whether a challenge is before a take on or vice versa
    # Statsbomb also seems to put tackles before take ons
    # Orders so that take ons always come before tackles

    # Create the next_* columns
    df['next_game'] = df['game_id'].shift(-1)
    df['next_period'] = df['period_id'].shift(-1)
    df['next_type_name'] = df['type_name'].shift(-1)
    df['next_team'] = df['team_id'].shift(-1)
    df['next_time'] = df['time_seconds'].shift(-1)

    # Find tackles that come before an opposition take on at the same timestamp
    df['action_id'] = df.apply(
        lambda row: row['action_id'] + 1.1 if (
            row['type_name'] == 'tackle' and 
            row['next_type_name'] == 'take_on' and 
            row['team_id'] != row['next_team'] and 
            row['time_seconds'] == row['next_time'] and 
            row['game_id'] == row['next_game'] and 
            row['period_id'] == row['next_period']
        ) else row['action_id'], axis=1
    )

    # Arrange by game_id, period_id, and action_id
    df = df.sort_values(by=['game_id', 'period_id', 'action_id'])

    # Reassign action_id to be sequential
    df['action_id'] = df.groupby('game_id').cumcount() + 1

    # Drop the next_* columns
    df = df.drop(columns=[col for col in df.columns if col.startswith('next_')])

    #reindex the dataframe now that rows have been reordered
    df = df.reset_index(drop=True)

    return df

def extra_interceptions(df):
    # Original code seems to miss interrupting actions following things like shots
    # This is meant to catch those edge cases in a last pass through the data

    # Create the prev_* columns
    df['prev_action_id'] = df['action_id'].shift(1)
    df['prev_action_type'] = df['type_name'].shift(1)
    df['prev_result_name'] = df['result_name'].shift(1)
    df['prev_team'] = df['team_id'].shift(1)
    df['prev_game'] = df['game_id'].shift(1)
    df['prev_period'] = df['period_id'].shift(1)
    df['prev_time'] = df['time_seconds'].shift(1)

    # Filter for actions where the prior action was the opposition
    extra_interception_info = df[
        (df['prev_game'] == df['game_id']) &
        (df['prev_period'] == df['period_id']) &
        (df['prev_team'] != df['team_id'])
    ].copy()

    # Filter out set piece and ball winning actions
    extra_interception_info = extra_interception_info[
        ~extra_interception_info['type_name'].isin([
            "interception", "tackle", "keeper_punch", "shot_penalty",
            "keeper_save", "keeper_claim", "keeper_pick_up", "throw_in", "freekick_short",
            "freekick_crossed", "shot_freekick", "corner_crossed", "corner_short",
            "clearance", "goalkick", "foul"
        ]) &
        (extra_interception_info['type_id'] != 24) &
        ~((extra_interception_info['prev_action_type'] == 'tackle') & (extra_interception_info['prev_result_name'] == 'fail')) &
        (extra_interception_info['type_name'] != 'bad_touch')
    ]

    # Impute a new action id matching socceraction convention
    extra_interception_info['action_id'] = extra_interception_info['action_id'] - 1 + 0.1
    extra_interception_info['result_id'] = 1

    extra_interception_info['type_id'] = 10
    extra_interception_info['type_name'] = 'interception'
    extra_interception_info['result_name'] = 'success'
    extra_interception_info['result_id'] = 1

    # Start and end coordinates should be the same
    extra_interception_info['x_end'] = extra_interception_info['x_start']
    extra_interception_info['y_end'] = extra_interception_info['y_start']

    # Combine the original DataFrame with the extra interception info
    out = pd.concat([df, extra_interception_info]).sort_values(by=['game_id', 'period_id', 'action_id'])

    # Reassign action_id to be sequential
    out['action_id'] = out.groupby('game_id').cumcount() + 1

    #Retain only desired columns
    out = out.drop(columns=[col for col in out.columns if col.startswith('prev_')])

    #reindex the dataframe now that rows have been added
    out = out.reset_index(drop=True)

    return out

def extra_out_result(df):
    # Adds rows for goal, own goal, and shot out of bounds outcomes
    # Equivalent to _extra_from_shots from original function

    # Create the next_* columns
    df['next_game'] = df['game_id'].shift(-1)
    df['next_period'] = df['period_id'].shift(-1)
    df['next_action'] = df['type_name'].shift(-1)

    # Mark shots that went out of bounds
    df['out'] = df['next_action'].isin(['corner_crossed', 'corner_short', 'goalkick']).astype(int)

    # Update result_name and result_id based on the out condition
    df['result_name'] = df.apply(
        lambda row: 'out' if row['out'] == 1 and row['next_game'] == row['game_id'] and row['next_period'] == row['period_id'] else row['result_name'],
        axis=1
    )
    df['result_id'] = df.apply(
        lambda row: 6 if row['out'] == 1 and row['next_game'] == row['game_id'] and row['next_period'] == row['period_id'] else row['result_id'],
        axis=1
    )

    # Drop the next_* and out columns
    df = df.drop(columns=['next_game', 'next_period', 'next_action', 'out'])

    return df

def remove_consecutive_interceptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove consecutive interception events by the same player,
    keeping only the first one in any consecutive run.
    """
    # 1. Build prev_* columns
    df['prev_game']    = df['game_id'].shift(1)
    df['prev_period']  = df['period_id'].shift(1)
    df['prev_player']  = df['player_id'].shift(1)
    df['prev_type']    = df['type_name'].shift(1)

    # 2. Identify interceptions to drop:
    #    - current is interception
    #    - previous is interception
    #    - same game, same period, same player
    drop_mask = (
        (df['type_name'] == 'interception') &
        (df['prev_type'] == 'interception') &
        (df['game_id'] == df['prev_game']) &
        (df['period_id'] == df['prev_period']) &
        (df['player_id'] == df['prev_player'])
    )

    # 3. Filter out the duplicates and clean up
    out = (
        df.loc[~drop_mask]
                 .drop(columns=[col for col in df.columns if col.startswith('prev_')])
                 .reset_index(drop=True)
    )

    # Reassign action_id to be sequential
    out['action_id'] = out.groupby('game_id').cumcount() + 1

    return out

def adjust_interception_clearance_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add result columns for interception events.
    """
    # 1. Build prev_* columns
    df['next_game']    = df['game_id'].shift(-1)
    df['next_period']  = df['period_id'].shift(-1)
    df['next_team']    = df['team_id'].shift(-1)
    df['next_type']    = df['type_name'].shift(-1)

    # 2. Identify interception result based on next action team, retaining any out results
    success_mask = (
        (df['type_name'].isin(['interception','clearance'])) &
        (df['next_team'] == df['team_id']) &
        (df['game_id'] == df['next_game']) &
        (df['period_id'] == df['next_period'])
    )
    fail_mask = (
        (df['type_name'].isin(['interception','clearance'])) &
        (df['next_team'] != df['team_id']) &
        (df['game_id'] == df['next_game']) &
        (df['period_id'] == df['next_period'])
    )
    out_mask = (
        (df['type_name'].isin(['interception','clearance'])) &
        (df['result_name'] == 'out')
    )

    # Assign results based on the masks
    df['result_name'] = np.where(success_mask, 'success', df['result_name'])
    df['result_name'] = np.where(fail_mask, 'fail', df['result_name'])
    df['result_name'] = np.where(out_mask, 'out', df['result_name'])
    df['result_id'] = np.where(success_mask, 1, df['result_id'])
    df['result_id'] = np.where(fail_mask, 0, df['result_id'])
    df['result_id'] = np.where(out_mask, 6, df['result_id'])

    # 3. Clean up
    df = df.drop(columns=['next_game', 'next_period', 'next_team', 'next_type'])

    return df

def impute_xy(df):
    # Impute x and y coordinates for events with missing values
    # Uses previous row values for imputation
    # Quick visual inspection of the data suggests that this is a reasonable approach
    df[['x_start', 'y_start']] = df[['x_start', 'y_start']].fillna(method='ffill')
    return df



def obscure_shot_result_add_goal_ontarget_columns (df):

    # Create goal and ontarget outcome columns
    df['goal'] = (
        ((df['type_name'].isin(['shot', 'shot_freekick', 'shot_penalty'])) & (df['result_name'] == 'success'))
        .astype(int)
    )

    df['ontarget'] = (
        ((df['type_name'].isin(['shot', 'shot_freekick', 'shot_penalty'])) & (df['result_name'].isin(['success', 'ontarget'])))
        .astype(int)
    )
    
    #Identify shots
    shot_rows = df['type_name'].isin(['shot', 'shot_freekick', 'shot_penalty'])

    # Change result of all shots to be special shot result
    # Obscures shot result from model
    # Retain the original result_id and result_name for restoring after model is run
    df['shot_original_result_id'] = df['result_id'].where(shot_rows).astype('Int64')
    df['shot_original_result_name'] = df['result_name'].where(shot_rows).astype('string')
    df.loc[shot_rows, 'result_id'] = 7
    df.loc[shot_rows, 'result_name'] = 'shot'

    # set all shots to have same end coordinates as start coordinates to obscure shot result
    # Retain the original end coordinates for restoring after model is run
    df['shot_original_x_end'] = df['x_end'].where(shot_rows).astype('float')
    df['shot_original_y_end'] = df['y_end'].where(shot_rows).astype('float')
    df.loc[shot_rows, 'x_end'] = df.loc[shot_rows, 'x_start']
    df.loc[shot_rows, 'y_end'] = df.loc[shot_rows, 'y_start']

    #reindex the dataframe now that rows have been added
    df = df.reset_index(drop=True)

    return df

def add_clearance_end_coords(df):
    # Adds end coordinates for clearances based on start coordinates of next action
    next_actions = df.shift(-1)
    next_actions[-1:] = df[-1:]
    clearance_idx = df["type_name"] == "clearance"
    df.loc[clearance_idx, 'x_end'] = df.loc[clearance_idx, 'x_start']
    df.loc[clearance_idx, 'y_end'] = df.loc[clearance_idx, 'y_start']

    return df

def fix_wyscout_events(df_events):
    """
    This function does some fixes on the Wyscout events such that the 
    spadl action dataframe can be built
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe with an extra column 'offside'
    """
    df_events = wyscout_create_shot_coordinates(df_events)
    df_events = wyscout_convert_duels(df_events)
    df_events = wyscout_insert_interception_passes(df_events)
    df_events = wyscout_add_offside_variable(df_events)
    df_events = wyscout_convert_touches(df_events)
    return df_events

def wyscout_create_shot_coordinates(df_events):
    """
    This function creates shot coordinates (estimates) from the Wyscout tags
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe with end coordinates for shots
    """
    goal_center_idx = (
        (df_events["shot.goalZone"]=='gt')
        | (df_events["shot.goalZone"]=='gc')
        | (df_events["shot.goalZone"]=='gb')
    )
    df_events.loc[goal_center_idx, "x_end"] = 100.0
    df_events.loc[goal_center_idx, "y_end"] = 50.0

    goal_right_idx = (
        (df_events["shot.goalZone"]=='grt')
        | (df_events["shot.goalZone"]=='gr')
        | (df_events["shot.goalZone"]=='grb')
    )
    df_events.loc[goal_right_idx, "x_end"] = 100.0
    df_events.loc[goal_right_idx, "y_end"] = 55.0

    goal_left_idx = (
        (df_events["shot.goalZone"]=='glt')
        | (df_events["shot.goalZone"]=='gl')
        | (df_events["shot.goalZone"]=='glb')
    )
    df_events.loc[goal_left_idx, "x_end"] = 100.0
    df_events.loc[goal_left_idx, "y_end"] = 45.0

    out_center_idx = (
        (df_events["shot.goalZone"]=='ot')
        | (df_events["shot.goalZone"]=='pt')
    )
    df_events.loc[out_center_idx, "x_end"] = 100.0
    df_events.loc[out_center_idx, "y_end"] = 50.0

    out_right_idx = (
        (df_events["shot.goalZone"]=='ort')
        | (df_events["shot.goalZone"]=='or')
        | (df_events["shot.goalZone"]=='orb')
    )
    df_events.loc[out_right_idx, "x_end"] = 100.0
    df_events.loc[out_right_idx, "y_end"] = 60.0

    out_left_idx = (
        (df_events["shot.goalZone"]=='olt')
        | (df_events["shot.goalZone"]=='ol')
        | (df_events["shot.goalZone"]=='olb')
    )
    df_events.loc[out_left_idx, "x_end"] = 100.0
    df_events.loc[out_left_idx, "y_end"] = 40.0

    post_left_idx = (
        (df_events["shot.goalZone"]=='plt')
        | (df_events["shot.goalZone"]=='pl')
        | (df_events["shot.goalZone"]=='plb')
    )
    df_events.loc[post_left_idx, "x_end"] = 100.0
    df_events.loc[post_left_idx, "y_end"] = 55.38

    post_right_idx = (
        (df_events["shot.goalZone"]=='prt')
        | (df_events["shot.goalZone"]=='pr')
        | (df_events["shot.goalZone"]=='prb')
    )
    df_events.loc[post_right_idx, "x_end"] = 100.0
    df_events.loc[post_right_idx, "y_end"] = 44.62

    blocked_idx = df_events["shot.goalZone"]=='bc'
    df_events.loc[blocked_idx, "x_end"] = df_events.loc[blocked_idx, "x_start"]
    df_events.loc[blocked_idx, "y_end"] = df_events.loc[blocked_idx, "y_start"]

    return df_events

def wyscout_convert_duels(df_events):
    """
    This function converts Wyscout duels that end with the ball out of field
    (subtype_id 50) into a pass for the player winning the duel to the location
    of where the ball went out of field. The remaining duels are removed as
    they are not on-the-ball actions.
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe in which the duels are either removed or transformed into a pass
    """

    # Shift events dataframe by one and two time steps
    df_events1 = df_events.shift(-1)
    df_events2 = df_events.shift(-2)

    # Define selector for same period id
    selector_same_period = df_events["period_id"] == df_events2["period_id"]

    # Define selector for duels that are followed by an 'out of field' event
    selector_duel_out_of_field = (
        (df_events["type.primary"] == 'duel')
        & (df_events1["type.primary"] == 'duel')
        & (df_events2["ball_out"] == 1)
        & selector_same_period
    )

    # Define selectors for current time step
    selector0_duel_won = selector_duel_out_of_field & (
        df_events["team_id"] != df_events2["team_id"]
    )
    selector0_duel_won_air = selector0_duel_won & (df_events["aerial_duel"] == 1)
    selector0_duel_won_not_air = selector0_duel_won & (df_events["aerial_duel"] != 1)

    # Define selectors for next time step
    selector1_duel_won = selector_duel_out_of_field & (
        df_events1["team_id"] != df_events2["team_id"]
    )
    selector1_duel_won_air = selector1_duel_won & (df_events1["aerial_duel"] == 1)
    selector1_duel_won_not_air = selector1_duel_won & (df_events1["aerial_duel"] != 1)

    # Aggregate selectors
    selector_duel_won = selector0_duel_won | selector1_duel_won
    selector_duel_won_air = selector0_duel_won_air | selector1_duel_won_air
    selector_duel_won_not_air = selector0_duel_won_not_air | selector1_duel_won_not_air

    # Set types and subtypes
    df_events.loc[selector_duel_won, "type.primary"] = 'pass'
    df_events.loc[selector_duel_won_air, "head_pass"] = 1

    # set end location equal to ball out of field location
    df_events.loc[selector_duel_won, "pass.accurate"] = False
    df_events.loc[selector_duel_won, "x_end"] = (
        100 - df_events2.loc[selector_duel_won, "x_start"]
    )
    df_events.loc[selector_duel_won, "y_end"] = (
        100 - df_events2.loc[selector_duel_won, "y_start"]
    )

    # Define selector for ground attacking duels with take on
    selector_attacking_duel = df_events["offensive_duel"] == 1
    selector_take_on = df_events["groundDuel.takeOn"]==1
    selector_att_duel_take_on = selector_attacking_duel & selector_take_on

    # Set take ons type
    df_events.loc[selector_att_duel_take_on, "type.primary"] = 'take_on'

    # Define selector for ground defensive duels that are tackles (take ons)
    selector_defensive_duel = df_events["defensive_duel"] == 1
    selector_def_duel_take_on = selector_defensive_duel & selector_take_on

    # Set tackle type
    df_events.loc[selector_def_duel_take_on, "type.primary"] = 'tackle'

    # Remove the remaining duels
    df_events = df_events[df_events["type.primary"] != 'duel']

    # Reset the index
    df_events = df_events.reset_index(drop=True)

    return df_events

def wyscout_insert_interception_passes(df_events):
    """
    This function converts passes (type_id 8) that are also interceptions
    (tag interception) in the Wyscout event data into two separate events,
    first an interception and then a pass.
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe in which passes that were also denoted as interceptions in the Wyscout
    notation are transformed into two events
    """

    df_events_interceptions = df_events[
        (df_events["interception"]==1) & (df_events["type.primary"] == 'pass')
    ].copy()

    if not df_events_interceptions.empty:
        #is it necessary to set everything to false/0? need a new way to do this without tag df that is outdated
        # df_events_interceptions.loc[:, [t[1] for t in wyscout_tags]] = False
        # df_events_interceptions["interception"] = True
        df_events_interceptions["type.primary"] = 'interception'
        df_events_interceptions[["x_end", "y_end"]] = df_events_interceptions[
            ["x_start", "y_start"]
        ]
        df_events_interceptions['milliseconds'] = df_events_interceptions['milliseconds'] - 1

        df_events = pd.concat([df_events_interceptions, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "milliseconds"])
        df_events = df_events.reset_index(drop=True)

    return df_events

def wyscout_add_offside_variable(df_events):
    """
    This function removes the offside events in the Wyscout event data and adds
    sets offside to 1 for the previous event (if this was a passing event)
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe with an extra column 'offside'
    """

    # Create a new column for the offside variable
    df_events["offside"] = 0

    # Shift events dataframe by one timestep
    df_events1 = df_events.shift(-1)

    # Select offside passes
    selector_offside = (df_events1["type.primary"] == 'offside') & (df_events["type.primary"] == 'pass')

    # Set variable 'offside' to 1 for all offside passes
    df_events.loc[selector_offside, "offside"] = 1

    # Remove offside events
    df_events = df_events[df_events["type.primary"] != 'offside']

    # Reset index
    df_events = df_events.reset_index(drop=True)

    return df_events

def wyscout_convert_touches(df_events):
    """
    This function converts the Wyscout 'touch' event (sub_type_id 72) into either
    a dribble or a pass (accurate or not depending on receiver)
    Args:
    df_events (pd.DataFrame): Wyscout event dataframe
    Returns:
    pd.DataFrame: Wyscout event dataframe without any touch events
    """

    df_events1 = df_events.shift(-1)

    selector_touch = (df_events["type.primary"] == 'touch') & ~(df_events["interception"]==1)

    selector_same_player = df_events["player_id"] == df_events1["player_id"]
    selector_same_team = df_events["team_id"] == df_events1["team_id"]

    #selector_touch_same_player = selector_touch & selector_same_player
    selector_touch_same_team = (
        selector_touch & ~selector_same_player & selector_same_team
    )
    selector_touch_other = selector_touch & ~selector_same_player & ~selector_same_team

    same_x = abs(df_events["x_end"] - df_events1["x_start"]) < min_dribble_length
    same_y = abs(df_events["y_end"] - df_events1["y_start"]) < min_dribble_length
    same_loc = same_x & same_y

    df_events.loc[selector_touch_same_team & same_loc, "type.primary"] = 'pass'
    df_events.loc[selector_touch_same_team & same_loc, "pass.accurate"] = True

    df_events.loc[selector_touch_other & same_loc, "type.primary"] = 'pass'
    df_events.loc[selector_touch_other & same_loc, "pass.accurate"] = False

    return df_events

def wyscout_fix_goalkick_coordinates(df_actions):
    """
    This function sets the goalkick start coordinates to (5,34)
    Args:
    df_actions (pd.DataFrame): SciSports action dataframe with
    start coordinates for goalkicks in the corner of the pitch
    Returns:
    pd.DataFrame: SciSports action dataframe including start coordinates for goalkicks
    """
    goalkicks_idx = df_actions["type_name"] == "goalkick"
    df_actions.loc[goalkicks_idx, "x_start"] = 5.0
    df_actions.loc[goalkicks_idx, "y_start"] = 34.0

    return df_actions

def wyscout_fix_keeper_save_coordinates(df_actions):
    """
    Original code inverts coordinates for all keeper_save actions
    It appears now that the coordinates are already in the correct format most of the time
    Some are incorrectly at x > 90 which has to be just from the wrong perspecive
    This function identifies these and inverts them for any keeper event
    Args:
    df_actions (pd.DataFrame): SciSports action dataframe with start coordinates in the corner of the pitch
    Returns:
    pd.DataFrame: SciSports action dataframe with correct keeper_save coordinates
    """
    inverted_keeper_events_idx = df_actions["type_name"].isin(['keeper_save', 'keeper_claim', 'keeper_punch']) & (df_actions["x_start"] > 80)    # invert the coordinates
    df_actions.loc[inverted_keeper_events_idx, "x_start"] = 105 - df_actions.loc[inverted_keeper_events_idx, "x_start"]
    df_actions.loc[inverted_keeper_events_idx, "y_start"] = 68 - df_actions.loc[inverted_keeper_events_idx, "y_start"]

    return df_actions

def wyscout_remove_keeper_goal_actions(df_actions):
    """
    This function removes keeper_save actions that appear directly after a goal
    Args:
    df_actions (pd.DataFrame): SciSports action dataframe with keeper actions directly after a goal
    Returns:
    pd.DataFrame: SciSports action dataframe without keeper actions directly after a goal
    """
    prev_actions = df_actions.shift(1)
    same_phase = prev_actions['time_seconds'] + 10 > df_actions['time_seconds']
    goals = (prev_actions['type_name'].isin(['shot','shot_freekick','shot_penalty'])) & (
        prev_actions['result_name'] == 'success'
    )
    keeper_saves = df_actions["type_name"] == "keeper_save"
    saves_after_goals_idx = same_phase & goals & keeper_saves
    df_actions = df_actions.drop(df_actions.index[saves_after_goals_idx])
    df_actions = df_actions.reset_index(drop=True)

    return df_actions

def wyscout_adjust_goalkick_result(df_actions):
    """
    This function adjusts goalkick results depending on whether
    the next action is performed by the same team or not
    This is part of the original code and I haven't confirmed that it's necessary now but it doesn't seem like it would hurt anything
    Args:
    df_actions (pd.DataFrame): SciSports action dataframe with incorrect goalkick results
    Returns:
    pd.DataFrame: SciSports action dataframe with correct goalkick results
    """
    next_actions = df_actions.shift(-1)
    goalkicks = df_actions["type_name"] == "goalkick"
    same_team = df_actions["team_id"] == next_actions["team_id"]

    accurate = same_team & goalkicks
    not_accurate = ~same_team & goalkicks
    #preserve goalkicks marked offside earlier
    goalkicks_offside = (df_actions["type_name"] == "goalkick") & (df_actions["result_name"] == "offside")

    df_actions.loc[accurate, "result_id"] = 1
    df_actions.loc[accurate, "result_name"] = 'success'
    df_actions.loc[not_accurate, "result_id"] = 0
    df_actions.loc[not_accurate, "result_name"] = 'fail'
    df_actions.loc[goalkicks_offside, "result_id"] = 2
    df_actions.loc[goalkicks_offside, "result_name"] = 'offside'

    return df_actions

def wyscout_add_kickoff_indicator(df_actions):
    """
    This function adds a kickoff indicator to the actions dataframe
    Args:
    df_actions (pd.DataFrame): SciSports action dataframe with kickoff events
    Returns:
    pd.DataFrame: SciSports action dataframe with kickoff indicator
    """
    # Create a new column for the kickoff indicator
    df_actions["kickoff"] = False

    # Shift events dataframe by one timestep
    df_actions_prev = df_actions.shift(1)

    same_game = df_actions["game_id"] == df_actions_prev["game_id"]
    same_period = df_actions["period_id"] == df_actions_prev["period_id"]
    goal_prev = df_actions_prev["type_name"].isin(['shot', 'shot_freekick', 'shot_penalty']) & (df_actions_prev["result_name"] == 'success')
    near_midfield = (
        (df_actions["x_start"] > 48)
        & (df_actions["x_start"] < 58)
        & (df_actions["y_start"] > 29)
        & (df_actions["y_start"] < 39)
    )    
    is_pass = df_actions["type_name"] == "pass"

    # Set the kickoff indicator to 1 when previous event is a goal and other conditions are met
    df_actions.loc[same_game & same_period & goal_prev & near_midfield & is_pass, "kickoff"] = True
    # Set the kickoff indicator to 1 when this is the first event of the game and other conditions are met
    df_actions.loc[~same_game & near_midfield & is_pass, "kickoff"] = True
    df_actions.loc[~same_period & near_midfield & is_pass, "kickoff"] = True

    return df_actions

def calculate_team_xg(
    data: pd.DataFrame,
    penalties: bool = True,
    num_actions: int = 30,
    possession_group: bool = False,
    xg_col: str = "xg"
) -> pd.DataFrame:
    """
    Adds xg_team and xg_possession to the SPADL DataFrame.
    - data: must include [game_id, period_id, team_id, possession_id, action_id, type_name, xg_col]
    - penalties: include penalty shots in xG calculations
    - num_actions: how many prior actions (of any type) to look back over (None == all, only valid when grouping by possession)
    - possession_group: if True, confine look-back window to same possession
    """
    # 1) sanity checks
    required = ["game_id","period_id","team_id","possession_id","action_id","type_name",xg_col]
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Missing required column: {c}")
    if num_actions is None and not possession_group:
        raise ValueError(
            "If num_actions=None, possession_group must be True "
            "so that the window covers the entire possession."
        )

    # 2) copy & sort
    df = data.copy().sort_values(["game_id","period_id","action_id"])

    # 3) build array of xG contributions for discounting:
    shot_types = {"shot","shot_freekick","shot_penalty"}
    is_shot = df["type_name"].isin(shot_types)
    if not penalties:
        is_shot &= df["type_name"] != "shot_penalty"

    xg_calc = np.zeros(len(df))
    xg_calc[is_shot] = df.loc[is_shot, xg_col].fillna(0.07).values #median xg
    df["_xg_calc"] = xg_calc

    # 4) group definitions
    group_cols = ["game_id","period_id","possession_id"] if possession_group else ["game_id","period_id"]

    def _compute_probs(group: pd.DataFrame) -> pd.Series:
        # assumes group is sorted by action_id
        n = len(group)
        xg_vals = group["_xg_calc"].values
        teams   = group["team_id"].values

        #  log(1 - xg) vector
        #  this is faster down the line than repeated 1-xg because you can take the cumulative sum to get conditional probabilities
        log1m = np.log1p(-xg_vals)

        # sliding-window start indices
        if num_actions is None:
            start_idx = np.zeros(n, dtype=int)
        else:
            start_idx = np.arange(n) - num_actions
            start_idx[start_idx < 0] = 0

        p_no = np.ones(n)
        for t in np.unique(teams):
            mask_t = (teams == t).astype(float)
            # prefix-sum of log1m * mask_t
            cs = np.concatenate(([0.0], np.cumsum(log1m * mask_t)))
            idxs = np.where(teams == t)[0]
            # product over window = exp( cs[i] - cs[start] )
            p_no[idxs] = np.exp(cs[idxs] - cs[start_idx[idxs]])

        return pd.Series(p_no, index=group.index)

    # 5) compute probabilities
    df["probability_shot_will_happen"] = (
        df
        .groupby(group_cols, group_keys=False, sort=False)
        .apply(_compute_probs)
    )

    # 6) compute xg_team only for shots
    df["xg_team"] = np.where(
        is_shot,
        df[xg_col].fillna(0) * df["probability_shot_will_happen"],
        np.nan
    )

    # 7) sum xg_team per possession
    df["xg_possession"] = (
        df
        .groupby(["game_id","period_id","team_id","possession_id"], sort=False)["xg_team"]
        .transform("sum")
    )

    # 8) cleanup and return
    return df.drop(columns=["_xg_calc"])

def format_event_data(data, in_format="ath_opta"):
    if in_format == "ath_opta":
        # Remove duplicate columns
        data = data.drop(columns=["playerId"], errors='ignore')

        # Function to convert camelCase to snake_case
        def convert_column_camel_to_snake(col_name):
            snake_case_col_name = re.sub(r'([A-Z])', r'_\1', col_name).lower().lstrip('_')
            return snake_case_col_name

        # Apply the conversion function to all column names
        data.columns = [convert_column_camel_to_snake(col) for col in data.columns]

        # Rename some vars for later
        data = data.rename(columns={
            "period": "period_id",
            "game_clock": "time_seconds",
            "short_name": "team",
            "opp_short_name": "opponent",
            "opp_team_id": "opponent_id",
            "game_date": "date",
            "league_name": "league"
        })

        data['season'] = data['season_name'].apply(lambda x: x[:4] + ("-" + x[8:9] if x[5:6] == "/" else ""))
        data['home'] = data['home'].apply(lambda x: "home" if x else "away")
        data['original_event_id'] = data.apply(lambda row: f"{row['game_id']}-{row['team_id']}-{row['period_id']}-{row['time_seconds']}-{row['event_id']}", axis=1)

        # Set some vars
        pitch_length = 1.05
        pitch_width = 0.68

        # spadl format except dribbles
        data['x_end'] = data['x_end'].fillna(data['x_start'])
        data['y_end'] = data['y_end'].fillna(data['y_start'])

        data['bodypart_name'] = np.where(data['header'] | data['head_pass'], 'head', 
                                       np.where(data['other_bodypart'], 'other', 'foot'))

        data['type_name'] = np.select(
            [
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['throw_in']),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['fk_taken'] & data['cross']),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & (data['fk_taken'] | data['kick_off'])),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['corner_taken'] & data['cross']),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['corner_taken']),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['cross']),
                (data['event_type'].isin(["Pass", "OffsidePass"]) & data['goal_kick']),
                (data['event_type'].isin(["Pass", "OffsidePass"])),
                (data['event_type'].isin(["TakeOn"])),
                (data['event_type'].isin(["FreeKick"]) & ~data['success']),
                (data['event_type'].isin(["Dismissal"])),
                (data['event_type'].isin(["Tackle", "Challenge"])),
                (data['event_type'].isin(["Interception", "BlockedPass"])),
                (data['event_type'].isin(["Miss", "Post", "AttemptSaved", "PenaltyGoal"]) & data['penalty']),
                (data['event_type'].isin(["Miss", "Post", "AttemptSaved", "Goal"]) & data['fk_taken']),
                (data['event_type'].isin(["Miss", "Post", "AttemptSaved", "Goal"])),
                (data['event_type'].isin(["Save"]) & (data['position'] == 'Goalkeeper')),
                (data['event_type'].isin(["Save"])),
                (data['event_type'].isin(["Claim"])),
                (data['event_type'].isin(["Punch"])),
                (data['event_type'].isin(["Pickup"])),
                (data['event_type'].isin(["Clearance"])),
                (data['event_type'].isin(["BallTouch", "OwnGoal", "Dispossessed"]))
            ],
            [
                "throw_in",
                "freekick_crossed",
                "freekick_short",
                "corner_crossed",
                "corner_short",
                "cross",
                "goalkick",
                "pass",
                "take_on",
                "foul",
                "foul",
                "tackle",
                "interception",
                "shot_penalty",
                "shot_freekick",
                "shot",
                "keeper_save",
                "interception",
                "keeper_claim",
                "keeper_punch",
                "keeper_pick_up",
                "clearance",
                "bad_touch"
            ],
            default="non_action"
        )

        #add result names
        data['result_name'] = np.select(
            [
                (data['event_type'] == "OffsidePass"),
                (data['event_type'] == "AttemptSaved") & (data['position'].shift(-1) == 'Goalkeeper'),
                (data['event_type'].isin(["Miss", "Post", "AttemptSaved"])),
                (data['event_type'] == "OwnGoal"),
                (data['event_type'] == "Goal"),
                (data['event_type'] == "Dismissal"),
                (data['event_type'] == "BallTouch"),
                (data['success'] == True)
            ],
            [
                "offside",
                "ontarget",
                "fail",
                "owngoal",
                "success",
                "red_card",
                "fail",
                "success"
            ],
            default="fail"
        )

        spadl = data[data['type_name'] != 'non_action'].copy()
        spadl = spadl.sort_values(by=['date', 'game_id', 'period_id', 'time_seconds', 'timestamp'])
        spadl['action_id'] = spadl.groupby('game_id').cumcount() + 1

        spadl = spadl.drop(columns=[
            "opta_team_id", "opp_opta_team_id", "full_name", "opp_full_name", "event_id", "league_country",
            "season_name", "season_id", "timestamp", "event_type", "success", "cross", "throw_in", "goal_kick",
            "corner_taken", "fk_taken", "kick_off", "penalty", "header", "head_pass", "other_bodypart", "team_long",
            "opponent_long", "league_id"
        ], errors='ignore')

        spadl['action_id'] = spadl['action_id'].astype(float)

        # derive carry events between events that meet dribble length distance qualifications
        spadl = spadl.sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl['dx'] = (spadl['x_start'].shift(-1) * pitch_length) - (spadl['x_end'] * pitch_length)
        spadl['dy'] = (spadl['y_start'].shift(-1) * pitch_width) - (spadl['y_end'] * pitch_width)
        # mike - combine distance filter with time, so dribbles that actually occur that are long in distance are included but not if they only take 2 seconds etc
        spadl['far_enough'] = (spadl['dx']**2 + spadl['dy']**2) >= min_dribble_length**2
        spadl['not_too_far'] = (spadl['dx']**2 + spadl['dy']**2) <= max_dribble_length**2
        spadl['dt'] = spadl['time_seconds'].shift(-1) - spadl['time_seconds']
        spadl['same_player'] = (spadl['receiver_id'].notna() & (spadl['receiver_id'] == spadl['player_id'].shift(-1))) | (spadl['player_id'] == spadl['player_id'].shift(-1))
        spadl['same_team'] = spadl['team_id'] == spadl['team_id'].shift(-1)
        spadl['same_phase'] = spadl['dt'] < max_dribble_duration
        spadl['same_period'] = spadl['period_id'] == spadl['period_id'].shift(-1)
        spadl['same_possession'] = spadl['possession_number'] == spadl['possession_number'].shift(-1)
        spadl['same_game'] = spadl['game_id'] == spadl['game_id'].shift(-1)
        spadl['not_stoppage'] = ~spadl['type_name'].shift(-1).isin(['corner_crossed', 'corner_short', 'shot_penalty', 'freekick_crossed', 'freekick_short', 'goalkick', 'throw_in'])
        spadl['dribble_idx'] = spadl['same_player'] & spadl['same_team'] & spadl['far_enough'] & spadl['not_too_far'] & spadl['same_phase'] & spadl['same_possession'] & spadl['same_period'] & spadl['same_game'] & spadl['not_stoppage']

        carries = spadl[spadl['dribble_idx'] | spadl['dribble_idx'].shift(1)].copy()
        carries['time_seconds'] = (carries['time_seconds'] + carries['time_seconds'].shift(-1)) / 2
        carries['elapsed_time'] = (carries['elapsed_time'] + carries['elapsed_time'].shift(-1)) / 2
        carries['team_id'] = carries['team_id'].shift(-1)
        carries['home'] = carries['home'].shift(-1)
        carries['player'] = carries['player'].shift(-1)
        carries['player_short'] = carries['player_short'].shift(-1)
        carries['position'] = carries['position'].shift(-1)
        carries['player_id'] = carries['player_id'].shift(-1)
        carries['receiver_id'] = np.nan
        carries['xg'] = np.nan
        carries['xa'] = np.nan
        carries['new_x_start'] = carries['x_end']
        carries['new_y_start'] = carries['y_end']
        carries['new_x_end'] = carries['x_start'].shift(-1)
        carries['new_y_end'] = carries['y_start'].shift(-1)

        carries = carries[carries['dribble_idx']].copy()
        carries = carries.drop(columns=['x_start', 'y_start', 'x_end', 'y_end'])
        carries = carries.rename(columns={
            "new_x_start": "x_start",
            "new_y_start": "y_start",
            "new_x_end": "x_end",
            "new_y_end": "y_end"
        })

        carries['action_id'] = carries['action_id'] + 0.2
        carries['possession_event'] = carries['possession_event'] + 0.2
        carries['bodypart_name'] = "foot"
        carries['type_name'] = "dribble"
        carries['result_name'] = "success"
        carries['receiver'] = None
        carries['receiver_short'] = None

        spadl = pd.concat([spadl, carries]).sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl = spadl.drop(columns=['dx', 'dy', 'far_enough', 'not_too_far', 'dt', 'same_player', 'same_team', 'same_phase', 'same_period', 'same_possession', 'same_game', 'dribble_idx'])

        #resize xy coordinates
        spadl['x_start'] = (pitch_length * spadl['x_start']).round(2)
        spadl['x_end'] = (pitch_length * spadl['x_end']).round(2)
        spadl['y_start'] = (pitch_width * spadl['y_start']).round(2)
        spadl['y_end'] = (pitch_width * spadl['y_end']).round(2)

        # Select the spadl columns only
        spadl = spadl[[
            'action_id', 'period_id', 'time_seconds','elapsed_time', 'type_name', 'result_name', 'bodypart_name', 'x_start', 'y_start', 'x_end', 'y_end', 'game_id', 'team_id', 'player_id', 'receiver_id', 'original_event_id','home','xg'
        ] ]

        #add spadl ids for results, bodyparts and action types
        spadl = spadl.merge(spadl_result_df, how='left', on='result_name')
        spadl = spadl.merge(spadl_action_df, how='left', on='type_name')
        spadl = spadl.merge(spadl_bodypart_df, how='left', on='bodypart_name')

        #reindex the dataframe now that rows have been dropped
        spadl = spadl.reset_index(drop=True)

        #reorder take ons and tackles so take ons always come first
        spadl = order_tackles(spadl)

        #add some extra interceptions not caught yet
        spadl = extra_interceptions(spadl)

        #change some results to out of bounds depending on next action
        spadl = extra_out_result(spadl)

        #add phases
        spadl = set_phase(spadl)

        #add a row for shots on target, location always center goal
        #also sets result of all shots to be shot and retains original result in new rows
        spadl = obscure_shot_result_add_goal_ontarget_columns(spadl)

    elif in_format == "statsbomb":

        # Function to convert camelCase to snake_case
        def convert_column_camel_to_snake(col_name):
            snake_case_col_name = re.sub(r'([A-Z])', r'_\1', col_name).lower().lstrip('_')
            return snake_case_col_name

        # Apply the conversion function to all column names
        data.columns = [convert_column_camel_to_snake(col) for col in data.columns]

        # Rename some vars for later
        data = data.rename(columns={
            "period": "period_id",
            "recipient_id": "receiver_id",
            "team_name": "team",
            "match_date": "date",
            "match_id": "game_id",
            "id" : "event_id",
            "end_x" : "x_end",
            "end_y" : "y_end",
            "start_x" : "x_start",
            "start_y" : "y_start",
            "league_name": "league"

        })

        #filter only for status==COMPLETE events
        #there seem to be some events with na status that get overridden by complete events later
        data = data[data['status'] == 'COMPLETE'].copy()
        data = data.reset_index(drop=True)

        data['time_seconds'] = data['minute'] * 60 + data['second']
                
        data['original_event_id'] = data.apply(lambda row: f"{row['game_id']}-{row['team_id']}-{row['period_id']}-{row['time_seconds']}-{row['event_id']}", axis=1)

        #mark kick-offs for phases
        #todo - consider these a separate event type
        data['kickoff'] = np.where(data['type'] == 'kick-off', True, False)

        # Add through_ball column - True if technique is through-ball, False otherwise
        data['through_ball'] = np.where(data['technique'] == 'through-ball', True, False)
        # Fill missing values in key_pass and assist with False, originally True/NA
        data['key_pass'] = data['key_pass'].fillna(False)
        data['assist'] = data['assist'].fillna(False)

        # Set some vars
        pitch_length = 0.875
        pitch_width = 0.85

        # spadl format except dribbles
        data['x_end'] = data['x_end'].fillna(data['x_start'])
        data['y_end'] = data['y_end'].fillna(data['y_start'])

        #impute some missing x and y coordinates
        data = impute_xy(data)

        #define crosses
        data = mark_crosses_sb(data)

        data['bodypart_name'] = np.where(data['body_part']=='head', 'head', 
                                       np.where(data['body_part']=='other', 'other', 
                                                np.where(data['body_part']=='left-foot', 'foot_left', 
                                                         np.where(data['body_part']=='right-foot', 'foot_right', 'foot'))))

        data['type_name'] = np.select(
            [
                (data['name'] == 'pass') & (data['type'] == 'throw-in'),
                (data['name'] == 'pass') & (data['type'] == 'free-kick') & ((data['cross']) | (data['height'] == 'high')),
                (data['name'] == 'pass') & ((data['type'] == 'free-kick') | (data['type'] == 'kick-off')),
                (data['name'] == 'pass') & (data['type'] == 'corner') & ((data['cross']) | (data['height'] == 'high')),
                (data['name'] == 'pass') & (data['type'] == 'corner'),
                (data['name'] == 'pass') & (data['cross']),
                (data['name'] == 'pass') & (data['through_ball']),
                (data['name'] == 'pass') & (data['type'] == 'goal-kick'),
                (data['name'] == 'pass'),
                (data['name'] == 'dribble'),
                (data['name'] == 'foul-committed'),
                (data['type'] == 'tackle') | (data['type'] == 'smother'),
                (data['name'] == 'block'),
                (data['name'] == 'shot') & (data['type']=='penalty'),
                (data['name'] == 'shot') & (data['type'] == 'free-kick'),
                (data['name'] == 'shot'),
                (data['name'].isin(['interception','ball-recovery'])),
                (data['name'] == 'goal-keeper') & (data['type'].isin(["penalty-saved", "save", "shot-saved", "shot-saved-off-target", "shot-saved-to-post"])),
                #mike - not all sweeper actions should be a claim, some are outside the box and should be an interception or clearance or something. can check by outcome later
                (data['name'] == 'goal-keeper') & (data['type'].isin(["collected", "keeper-sweeper"])),
                (data['type'] == 'punch'),
                (data['type'] == 'smother'),
                (data['name'] == 'clearance'),
                (data['name'].isin(["miscontrol", "own-goal-against", "dispossessed"])),
                (data['name'] == 'dribbled-past'),
            ],
            [
                "throw_in",
                "freekick_crossed",
                "freekick_short",
                "corner_crossed",
                "corner_short",
                "cross",
                "through_ball",
                "goalkick",
                "pass",
                "take_on",
                "foul",
                "tackle",
                "interception",
                "shot_penalty",
                "shot_freekick",
                "shot",
                "interception",
                "keeper_save",
                "keeper_claim",
                "keeper_punch",
                "keeper_pick_up",
                "clearance",
                "bad_touch",
                "tackle"
            ],
            default="non_action"
        )

            #add result names
        data['result_name'] = np.select(
            [
                (data['outcome'] == "pass-offside"),
                (data['outcome'].isin(["saved", "saved-to-post"])),
                #is fail real or just a typo in docs?
                (data['outcome'].isin(["blocked", "wayward", "off-t", "saved-off-t", "fail"])),
                (data['name'] == "own-goal-against"),
                (data['card'].isin(["red-card", "second-yellow-card"])),
                (data['card'] == "yellow-card"),
                (data['name'] == "miscontrol"),
                (data['outcome'].isin(["complete", "goal", "success-in-play", "success-out", "success", "won", "claim", "clear", "punched-out", "saved-twice", "success-in-play", "touched-out"])),
                (data['name'] == "dribbled-past"),
            ],
            [
                "offside",
                "ontarget",
                "fail",
                "owngoal",
                "red_card",
                "yellow_card",
                "fail",
                "success",
                "fail"
            ],
            default="fail"
        )

        spadl = data[data['type_name'] != 'non_action'].copy()
        spadl = spadl.sort_values(by=['date', 'game_id', 'period_id', 'index'])
        spadl['action_id'] = spadl.groupby('game_id').cumcount() + 1

        spadl['action_id'] = spadl['action_id'].astype(float)

        # derive carry events between events that meet dribble length distance qualifications
        spadl = spadl.sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl['dx'] = (spadl['x_start'].shift(-1) * pitch_length) - (spadl['x_end'] * pitch_length)
        spadl['dy'] = (spadl['y_start'].shift(-1) * pitch_width) - (spadl['y_end'] * pitch_width)
        # mike - combine distance filter with time, so dribbles that actually occur that are long in distance are included but not if they only take 2 seconds etc
        spadl['far_enough'] = (spadl['dx']**2 + spadl['dy']**2) >= min_dribble_length**2
        spadl['not_too_far'] = (spadl['dx']**2 + spadl['dy']**2) <= max_dribble_length**2
        spadl['dt'] = spadl['time_seconds'].shift(-1) - spadl['time_seconds']
        spadl['same_player'] = (spadl['receiver_id'].notna() & (spadl['receiver_id'] == spadl['player_id'].shift(-1))) | (spadl['player_id'] == spadl['player_id'].shift(-1))
        spadl['same_team'] = spadl['team_id'] == spadl['team_id'].shift(-1)
        spadl['same_phase'] = spadl['dt'] < max_dribble_duration
        spadl['same_period'] = spadl['period_id'] == spadl['period_id'].shift(-1)
        spadl['same_game'] = spadl['game_id'] == spadl['game_id'].shift(-1)
        spadl['not_stoppage'] = ~spadl['type_name'].shift(-1).isin(['corner_crossed', 'corner_short', 'shot_penalty', 'freekick_crossed', 'freekick_short', 'goalkick', 'throw_in'])
        spadl['dribble_idx'] = spadl['same_player'] & spadl['same_team'] & spadl['far_enough'] & spadl['not_too_far'] & spadl['same_phase'] & spadl['same_period'] & spadl['same_game'] & spadl['not_stoppage']

        carries = spadl[spadl['dribble_idx'] | spadl['dribble_idx'].shift(1)].copy()
        carries['time_seconds'] = (carries['time_seconds'] + carries['time_seconds'].shift(-1)) / 2
        carries['team_id'] = carries['team_id'].shift(-1)
        carries['home'] = carries['home'].shift(-1)
        carries['player_id'] = carries['player_id'].shift(-1)
        carries['receiver_id'] = np.nan
        carries['xg'] = np.nan
        carries['new_x_start'] = carries['x_end']
        carries['new_y_start'] = carries['y_end']
        carries['new_x_end'] = carries['x_start'].shift(-1)
        carries['new_y_end'] = carries['y_start'].shift(-1)

        carries = carries[carries['dribble_idx']].copy()
        carries = carries.drop(columns=['x_start', 'y_start', 'x_end', 'y_end'])
        carries = carries.rename(columns={
            "new_x_start": "x_start",
            "new_y_start": "y_start",
            "new_x_end": "x_end",
            "new_y_end": "y_end"
        })

        carries['action_id'] = carries['action_id'] + 0.2
        carries['bodypart_name'] = "foot"
        carries['type_name'] = "dribble"
        carries['result_name'] = "success"

        spadl = pd.concat([spadl, carries]).sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl = spadl.drop(columns=['dx', 'dy', 'far_enough', 'not_too_far', 'dt', 'same_player', 'same_team', 'same_phase', 'same_period', 'same_game', 'dribble_idx'])

        #resize xy coordinates
        spadl['x_start'] = (pitch_length * spadl['x_start']).round(2)
        spadl['x_end'] = (pitch_length * spadl['x_end']).round(2)
        spadl['y_start'] = (pitch_width * spadl['y_start']).round(2)
        spadl['y_end'] = (pitch_width * spadl['y_end']).round(2)

        # Ensure position field is preserved from original data
        if 'position' in spadl.columns:
            spadl['position'] = spadl['position']
        else:
            spadl['position'] = ''

            # Select the spadl columns only
        spadl = spadl[[
            "action_id", "period_id", "time_seconds", "elapsed_time","type_name", "result_name", "bodypart_name",
            "x_start", "y_start", "x_end", "y_end", "game_id", "team_id", "player_id", "receiver_id", 
            "original_event_id", "home", "xg", "height", "through_ball", "position", "kickoff","key_pass","assist"
        ] ]

        #add spadl ids for results, bodyparts and action types
        spadl = spadl.merge(spadl_result_df, how='left', on='result_name')
        spadl = spadl.merge(spadl_action_df, how='left', on='type_name')
        spadl = spadl.merge(spadl_bodypart_df, how='left', on='bodypart_name')

        #reindex the dataframe now that rows have been dropped
        spadl = spadl.reset_index(drop=True)

        #add end coordinates for clearances
        spadl = add_clearance_end_coords(spadl)

        #reorder take ons and tackles so take ons always come first
        spadl = order_tackles(spadl)

        #add some extra interceptions not caught yet
        spadl = extra_interceptions(spadl)

        #change some results to out of bounds depending on next action
        spadl = extra_out_result(spadl)
        
        #remove any consecutive interceptions
        spadl = remove_consecutive_interceptions(spadl)

        #adjust interception and clearance results
        spadl = adjust_interception_clearance_result(spadl)

        #add phases
        spadl = set_phase(spadl)

        #add columns for goal and shot on target results, then change shot end coordinates and results to obscure result
        #prevents info leakage in model about result of shot but retains target goal information
        spadl = obscure_shot_result_add_goal_ontarget_columns(spadl)

        #add sequence and possession ids
        spadl = add_possession_sequence_ids(spadl)
    
    elif in_format == "wyscout":

        # Event locations
        data['x_start']=data['location.x']
        data['y_start']=data['location.y']
        data['x_end']=data['x_start']
        data['y_end']=data['y_start']
        data['x_end'] = np.where(data['pass.endLocation.x'].notnull(), data['pass.endLocation.x'], data['x_end'])
        data['y_end'] = np.where(data['pass.endLocation.y'].notnull(), data['pass.endLocation.y'], data['y_end'])

        # Rename some vars for later
        data["period_id"] = data['matchPeriodNum']
        data["event_id"] = data['id']
        data["player_id"] = data["player.id"]
        data['receiver_id'] = data["pass.recipient.id"]
        data['height'] = data['pass.height']
        data["team_id"] = data["team.id"]
        data["team_name"] = data["team.name"]
        data["game_id"] = data["matchId"]
        data['time_seconds'] = data['minute'] * 60 + data['second']
        data["milliseconds"] = data['time_seconds'] * 1000
        data['through_ball']=np.where(data['through_pass'] == 1, True, False)
        data['position']=data['player.position']
        data['xg'] = data['shot.xg']

        #reorder before event fixes
        data = data.sort_values(by=['game_id', 'period_id', 'matchTimestamp'])

        #apply wyscout specific event fixes
        data = fix_wyscout_events(data)

        #apply our own cross definition
        data['cross']=False
        data = mark_crosses_wyscout(data)

        data['original_event_id'] = data.apply(lambda row: f"{row['game_id']}-{row['team_id']}-{row['period_id']}-{row['time_seconds']}-{row['event_id']}", axis=1)

        # Set some vars
        pitch_length = 1.05
        pitch_width = 0.68

        #body part
        # Initialize the bodypart_name column with the default value "foot"
        data["bodypart_name"] = "foot"

        # Apply conditions to determine the body part
        data.loc[data["head_pass"] == 1, "bodypart_name"] = "head"
        data.loc[data["head_shot"] == 1, "bodypart_name"] = "head"
        data.loc[data["shot.bodyPart"] == "head_or_other", "bodypart_name"] = "other"
        data.loc[data["hand_pass"] == 1, "bodypart_name"] = "other"
        data.loc[data["type.primary"] == "throw_in", "bodypart_name"] = "other"
        data.loc[data["save"] == 1, "bodypart_name"] = "other"
        data.loc[data["infraction.type"] == "hand_foul", "bodypart_name"] = "other"
 
        # action type
        data["type_name"] = np.select(
            [
                (data["type.primary"] == "throw_in"),
                (data["type.primary"] == "free_kick") & (data["pass.height"] == "high"),
                (data["free_kick_shot"] == 1),
                (data["type.primary"] == "free_kick"),
                (data["type.primary"] == "corner") & (data["pass.height"] == "high"),
                (data["type.primary"] == "corner"),
                (data["type.primary"] == "goal_kick"),
                (data["type.primary"] == "pass") & (data['cross']==True),
                (data["type.primary"] == "pass") & (data['through_ball'] == True),
                (data["type.primary"] == "pass"),
                (data["type.primary"] == "take_on"),
                (data["type.primary"] == "infraction"),
                (data["type.primary"] == "tackle"),
                (data["type.primary"] == "interception"),
                (data["type.primary"] == "penalty"),
                (data["type.primary"] == "shot"),
                (data["save"] == 1),
                (data["type.primary"] == "clearance"),
                (data["type.primary"] == "touch") & (data["loss"] == 1),
                (data["type.primary"] == "own_goal")
            ],
            [
                "throw_in",
                "freekick_crossed",
                "shot_freekick",
                "freekick_short",
                "corner_crossed",
                "corner_short",
                "goalkick",
                "cross",
                "through_ball",
                "pass",
                "take_on",
                "foul",
                "tackle",
                "interception",
                "shot_penalty",
                "shot",
                "keeper_save",
                "clearance",
                "bad_touch",
                "bad_touch"
            ],
            default="non_action"
        )

            #add result names
        data["result_name"] = np.select(
            [
                (data["offside"] == 1),
                (data["type_name"] == "foul") & (data["infraction.yellowCard"] == True),
                (data["type_name"] == "foul") & (data["infraction.redCard"] == True),
                (data["type.primary"] == "own_goal"),
                (data["type_name"] == "foul"),
                (data["type.primary"] == "shot") & (data["shot.onTarget"] == True),
                (data["type.primary"] == "shot") & (data["goal"] == 1),
                (data["type.primary"] == "shot"),
                (data["type.primary"] == "penalty") & (data["goal"] == 1),
                (data["type.primary"] == "penalty"),
                (data["type.primary"].isin(["corner", "throw_in", "goal_kick", "pass", "free_kick"])) & (data["pass.accurate"] == True),
                (data["type.primary"].isin(["corner", "throw_in", "goal_kick", "pass", "free_kick"])),
                (data["type_name"].isin(["interception", "clearance", "keeper_save"])),
                (data["type_name"] == "bad_touch"),
                (data["type_name"] == "take_on") & (data["groundDuel.keptPossession"] == True),
                (data["type_name"] == "take_on"),
                (data["type_name"] == "tackle") & (data["groundDuel.recoveredPossession"] == True),
                (data["type_name"] == "tackle")
            ],
            [
                "offside",
                "yellow_card",
                "red_card",
                "owngoal",
                "fail",
                "ontarget",
                "success",
                "fail",
                "success",
                "fail",
                "success",
                "fail",
                "success",
                "fail",
                "success",
                "fail",
                "success",
                "fail"
            ],
            default="success"  # Default case but should not really be anything else
        )

        spadl = data[data['type_name'] != 'non_action'].copy()
        spadl = spadl.sort_values(by=['game_id', 'period_id', 'matchTimestamp'])
        spadl['action_id'] = range(1, len(spadl) + 1)

        spadl['action_id'] = spadl['action_id'].astype(float)

        # derive carry events between events that meet dribble length distance qualifications
        spadl = spadl.sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl['dx'] = (spadl['x_start'].shift(-1) * pitch_length) - (spadl['x_end'] * pitch_length)
        spadl['dy'] = (spadl['y_start'].shift(-1) * pitch_width) - (spadl['y_end'] * pitch_width)
        # mike - combine distance filter with time, so dribbles that actually occur that are long in distance are included but not if they only take 2 seconds etc
        spadl['far_enough'] = (spadl['dx']**2 + spadl['dy']**2) >= min_dribble_length**2
        spadl['not_too_far'] = (spadl['dx']**2 + spadl['dy']**2) <= max_dribble_length**2
        spadl['dt'] = spadl['time_seconds'].shift(-1) - spadl['time_seconds']
        spadl['same_player'] = (spadl['receiver_id'].notna() & (spadl['receiver_id'] == spadl['player_id'].shift(-1))) | (spadl['player_id'] == spadl['player_id'].shift(-1))
        spadl['same_team'] = spadl['team_id'] == spadl['team_id'].shift(-1)
        spadl['same_phase'] = spadl['dt'] < max_dribble_duration
        spadl['same_period'] = spadl['period_id'] == spadl['period_id'].shift(-1)
        spadl['same_game'] = spadl['game_id'] == spadl['game_id'].shift(-1)
        spadl['not_stoppage'] = ~spadl['type_name'].shift(-1).isin(['corner_crossed', 'corner_short', 'shot_penalty', 'freekick_crossed', 'freekick_short', 'goalkick', 'throw_in'])
        spadl['dribble_idx'] = spadl['same_player'] & spadl['same_team'] & spadl['far_enough'] & spadl['not_too_far'] & spadl['same_phase'] & spadl['same_period'] & spadl['same_game'] & spadl['not_stoppage']

        carries = spadl[spadl['dribble_idx'] | spadl['dribble_idx'].shift(1)].copy()
        carries['time_seconds'] = (carries['time_seconds'] + carries['time_seconds'].shift(-1)) / 2
        carries['team_id'] = carries['team_id'].shift(-1)
        carries['home'] = carries['home'].shift(-1)
        carries['player_id'] = carries['player_id'].shift(-1)
        carries['receiver_id'] = np.nan
        carries['xg'] = np.nan
        carries['new_x_start'] = carries['x_end']
        carries['new_y_start'] = carries['y_end']
        carries['new_x_end'] = carries['x_start'].shift(-1)
        carries['new_y_end'] = carries['y_start'].shift(-1)

        carries = carries[carries['dribble_idx']].copy()
        carries = carries.drop(columns=['x_start', 'y_start', 'x_end', 'y_end'])
        carries = carries.rename(columns={
            "new_x_start": "x_start",
            "new_y_start": "y_start",
            "new_x_end": "x_end",
            "new_y_end": "y_end"
        })

        carries['action_id'] = carries['action_id'] + 0.2
        carries['bodypart_name'] = "foot"
        carries['type_name'] = "dribble"
        carries['result_name'] = "success"

        spadl = pd.concat([spadl, carries]).sort_values(by=['game_id', 'period_id', 'action_id'])
        spadl = spadl.drop(columns=['dx', 'dy', 'far_enough', 'not_too_far', 'dt', 'same_player', 'same_team', 'same_phase', 'same_period', 'same_game', 'dribble_idx'])

        #resize xy coordinates
        spadl['x_start'] = (pitch_length * spadl['x_start']).round(2)
        spadl['x_end'] = (pitch_length * spadl['x_end']).round(2)
        spadl['y_start'] = (pitch_width * spadl['y_start']).round(2)
        spadl['y_end'] = (pitch_width * spadl['y_end']).round(2)

            # Select the spadl columns only
        spadl = spadl[[
            "action_id", "period_id", "time_seconds", "elapsed_time","type_name", "result_name", "bodypart_name",
            "x_start", "y_start", "x_end", "y_end", "game_id", "team_id", "player_id", "receiver_id", 
            "original_event_id", "home", "xg", "height", "through_ball", "position"
        ] ]

        #add spadl ids for results, bodyparts and action types
        spadl = spadl.merge(spadl_result_df, how='left', on='result_name')
        spadl = spadl.merge(spadl_action_df, how='left', on='type_name')
        spadl = spadl.merge(spadl_bodypart_df, how='left', on='bodypart_name')

        #reindex the dataframe now that rows have been dropped
        spadl = spadl.reset_index(drop=True)

        #wyscout keeper stuff makes no sense
        spadl = wyscout_fix_goalkick_coordinates(spadl)
        spadl = wyscout_adjust_goalkick_result(spadl)
        spadl = wyscout_fix_keeper_save_coordinates(spadl)
        spadl = wyscout_remove_keeper_goal_actions(spadl)

        #add kickoff column - needs to be done after remove_keeper_goal_actions
        #this allows the function to recognize the first pass afer a goal as a kickoff
        #otherwise theres a ghost save between the two events
        spadl = wyscout_add_kickoff_indicator(spadl)

        #add end coordinates for clearances
        spadl = add_clearance_end_coords(spadl)

        #reorder take ons and tackles so take ons always come first
        spadl = order_tackles(spadl)

        #add some extra interceptions not caught yet
        spadl = extra_interceptions(spadl)

        #change some results to out of bounds depending on next action
        spadl = extra_out_result(spadl)

        #remove any consecutive interceptions
        spadl = remove_consecutive_interceptions(spadl)

        #adjust interception and clearance results
        spadl = adjust_interception_clearance_result(spadl)

        #add phases - done before shot_ontarget so those events can inherit the phase from the shot
        spadl = set_phase(spadl)

        #add a column for goal and shot on target results, then change shot end coordinates and results to obscure result
        #prevents info leakage in model about result of shot but retains target goal information
        spadl = obscure_shot_result_add_goal_ontarget_columns(spadl)

        #add sequence and possession ids
        spadl = add_possession_sequence_ids(spadl)
    
    #calculate team xg and possession xg
    spadl = calculate_team_xg(spadl, num_actions=30, possession_group=False)
    #add source format for related model feature
    spadl['source'] = in_format

    return spadl

def set_phase(data):
    
    data = data.sort_values(by=['game_id', 'action_id'])


    # Extract columns as numpy arrays
    type_name = data['type_name'].values
    position = data['position'].values
    kickoff = data['kickoff'].values
    bodypart = data['bodypart_name'].values
    x_start = data['x_start'].values
    y_start = data['y_start'].values
    x_end = data['x_end'].values
    y_end = data['y_end'].values
    time_seconds = data['time_seconds'].values
    period_id = data['period_id'].values
    game_id = data['game_id'].values
    type_name = data['type_name'].values
    height = data['height'].values
    team_id = data['team_id'].values
    home=data['home'].values
    future_attacking_freekick_trigger=False #used to look ahead 5s for free kicks that enter box after 5s but convert to a different phase first
    future_attacking_throw_in_trigger=False #used to look ahead 5s for throw ins that enter box after 5s but convert to a different phase first

    #calculate velocity over last 5 seconds
    static_x_start = x_start.copy()
    static_x_start[home == 'away'] = 106 - static_x_start[home == 'away']
    static_y_start = y_start.copy()
    static_y_start[home == 'away'] = 68 - static_y_start[home == 'away']
    velocity_5s_start = np.full(len(data), np.nan)
    velocity_5s_start_y = np.full(len(data), np.nan)
    velocity_5s_start_homegoaldist = np.full(len(data), np.nan)
    velocity_5s_start_awaygoaldist = np.full(len(data), np.nan)
    velocity_5s_homegoaldist = np.full(len(data), np.nan)
    velocity_5s_awaygoaldist = np.full(len(data), np.nan)
    velocity_idx = np.full(len(data), np.nan)
    velocity_5s_chg = np.zeros(len(data))
    velocity_5s_timediff = np.zeros(len(data))
    velocity_5s_prog = np.full(len(data), np.nan)
    velocity_5s_prog_chg_home = np.zeros(len(data))
    velocity_5s_prog_perc_chg_home = np.zeros(len(data))
    velocity_5s_prog_chg_away = np.zeros(len(data))
    velocity_5s_prog_perc_chg_away = np.zeros(len(data))
    velocity_5s_prog_perc_chg = np.zeros(len(data))
    velocity_5s_prog_perc = np.zeros(len(data))
    velocity_5s = np.zeros(len(data))
    velocity_5s_homegoaldist = np.sqrt((static_x_start - 106)**2 + (static_y_start - 68/2)**2)
    velocity_5s_awaygoaldist = np.sqrt((static_x_start - 0)**2 + (static_y_start - 68/2)**2)
    time_middle_third_start = np.full(len(data), None)
    for i in range(len(data)):

        #mark time since last free kick to mark attacking free kicks that enter box after 5 seconds but convert to a different phase first


        # below are velocity 5s calculations
        target_time = time_seconds[i] - 5
        # Find the index of the closest time less than or equal to target_time
        valid_indices = np.where((time_seconds < time_seconds[i]) & (type_name != 'dribble') & (period_id==period_id[i]) & (game_id==game_id[i]))[0]
        if len(valid_indices) > 0:
            closest_idx = valid_indices[np.argmin(np.abs(time_seconds[valid_indices] - target_time))]
            velocity_idx[i]=closest_idx
            velocity_5s_start[i] = static_x_start[closest_idx]
            velocity_5s_start_y[i] = static_y_start[closest_idx]
            velocity_5s_start_homegoaldist[i] = velocity_5s_homegoaldist[closest_idx]
            velocity_5s_start_awaygoaldist[i] = velocity_5s_awaygoaldist[closest_idx]
            velocity_5s_chg[i] = static_x_start[i] - static_x_start[closest_idx]
            velocity_5s_prog_chg_home[i] = velocity_5s_start_homegoaldist[i] - velocity_5s_homegoaldist[i]
            velocity_5s_prog_perc_chg_home[i] = velocity_5s_prog_chg_home[i]/velocity_5s_start_homegoaldist[i]
            velocity_5s_prog_chg_away[i] = velocity_5s_start_awaygoaldist[i] - velocity_5s_awaygoaldist[i]
            velocity_5s_prog_perc_chg_away[i] = velocity_5s_prog_chg_away[i]/velocity_5s_start_awaygoaldist[i]
            velocity_5s_timediff[i] = time_seconds[i] - time_seconds[closest_idx]
            velocity_5s[i] = velocity_5s_chg[i]/velocity_5s_timediff[i]
            if home[i] == 'home':
                velocity_5s_prog[i] = velocity_5s_prog_chg_home[i]/velocity_5s_timediff[i]
                velocity_5s_prog_perc[i] = velocity_5s_prog_perc_chg_home[i]/velocity_5s_timediff[i]
                velocity_5s_prog_perc_chg[i] = velocity_5s_prog_perc_chg_home[i]
            elif home[i] == 'away':
                velocity_5s_prog[i] = velocity_5s_prog_chg_away[i]/velocity_5s_timediff[i]
                velocity_5s_prog_perc[i] = velocity_5s_prog_perc_chg_away[i]/velocity_5s_timediff[i]
                velocity_5s_prog_perc_chg[i] = velocity_5s_prog_perc_chg_away[i]

    velocity_5s[home == 'away'] = -1*velocity_5s[home == 'away']
    velocity_5s = np.nan_to_num(velocity_5s, nan=0.0, posinf=0.0, neginf=0.0)
    velocity_5s_chg[home == 'away'] = -1*velocity_5s_chg[home == 'away']
    velocity_5s_chg = np.nan_to_num(velocity_5s_chg, nan=0.0, posinf=0.0, neginf=0.0)

    #distances between two actions for loose balls
    prev_static_x=np.roll(static_x_start,1)
    prev_static_y=np.roll(static_y_start,1)
    distance_prev = np.sqrt((static_x_start - prev_static_x)**2 + (static_y_start - prev_static_y)**2)
    distance_prev[0] = 0
    goaldist_prev = np.zeros(len(home))
    velocity_5s_homegoaldist_prev = np.roll(velocity_5s_homegoaldist, 1)
    velocity_5s_awaygoaldist_prev = np.roll(velocity_5s_awaygoaldist, 1)
    velocity_5s_homegoaldist_prev[0] = 0
    velocity_5s_awaygoaldist_prev[0] = 0
    goaldist_prev = np.where(home == 'home', velocity_5s_homegoaldist_prev, velocity_5s_awaygoaldist_prev)



    # Initialize other new variables as numpy arrays
    phase = np.full(len(type_name), '', dtype='<U50')
    phase_start = np.zeros(len(data))
    phase_team = np.full(len(data), data['team_id'].iloc[0])
    phase_start_x = np.zeros(len(data))
    phase_start_y = np.zeros(len(data))
    phase_start_type_name = np.full(len(type_name), '', dtype='<U50')
    phase_time = np.zeros(len(data))
    phase_total_distance = np.zeros(len(data))
    phase_id = np.zeros(len(data))
    entered_box_phase = np.full(len(data), False)
    prog_action_phase = np.full(len(data), False)
    force_new_phase = np.full(len(data), False)
    securing_to_other_transition = np.full(len(data), False)
    distance = np.sqrt((x_start - x_end)**2 + (y_start - y_end)**2)
    start_box_ind = (y_start >= 14.35) & (y_start <= 53.65) & (x_start > 87.15)
    end_box_ind = (y_end >= 14.35) & (y_end <= 53.65) & (x_end > 87.15)
    start_own_box_ind = (y_start >= 14.35) & (y_start <= 53.65) & (x_start < 18.85)
    end_own_box_ind = (y_end >= 14.35) & (y_end <= 53.65) & (x_end < 18.85)
    indirect_free_kick_zone = (y_start >= 16.35) & (y_start <= 51.65) & (x_start > 89.15)
    start_goal_dist = np.sqrt((x_start - 106)**2 + (y_start - (68/2))**2)
    end_goal_dist = np.sqrt((x_end - 106)**2 + (y_end - (68/2))**2)
    prog_dist = start_goal_dist - end_goal_dist
    prog_action = prog_dist > .20*start_goal_dist
    prog_perc = prog_dist/start_goal_dist
    prog_dist_prev = goaldist_prev - start_goal_dist
    prog_perc_prev = np.zeros(len(data))
    prog_perc_prev[goaldist_prev != 0] = prog_dist_prev[goaldist_prev != 0] / goaldist_prev[goaldist_prev != 0]
    possaction = np.isin(type_name, ['pass', 'cross', 'dribble', 'take_on', 'shot'])
    defaction = np.isin(type_name, ['interception', 'clearance', 'tackle',
                                    'keeper_punch', 'keeper_save', 'keeper_catch','keeper_pick_up','keeper_claim'])
    high_ball_action = np.where((~np.isin(bodypart, ['foot', 'foot_left','foot_right'])) | (np.isin(type_name, ['clearance', 'keeper_punch'])) | ((height=='high') & np.isin(type_name, ['freekick_short', 'pass'])), True, False)
    loose_ball_action = np.where((np.isin(type_name, ['interception', 'clearance', 'tackle', 'foul'])) |
                                    (type_name=='bad_touch'), True, False)
    corner_short = np.where((type_name=='corner_short') & (~end_box_ind), True, False)
    corner_long = np.where((np.isin(type_name, ['corner_short', 'corner_crossed'])) & (~corner_short), True, False)
    pass_from_onetouch_interception = np.full(len(data), True)
    pass_from_onetouch_interception[1:] = (
        (x_start[1:] == x_start[:-1]) & 
        (y_start[1:] == y_start[:-1]) & 
        (time_seconds[1:] == time_seconds[:-1])  & (team_id[1:] == team_id[:-1]) &
        (type_name[1:] == 'pass') &
        (type_name[:-1] == 'interception')
    )
    interception_from_onetouch_pass = np.roll(pass_from_onetouch_interception, -1)
    interception_from_onetouch_pass[-1] = False


    # num_passes_phase = 0
    num_possaction_phase = 0
    

    # Loop through each row
    for i in range(len(data)):

        # Set the value of phase and phase start to the previous row's value of 'phase'
        if i > 0:
            phase[i] = phase[i-1].replace('defending_','')
            phase_id[i] = phase_id[i-1]
            phase_start[i] = phase_start[i-1]
            phase_time[i] = time_seconds[i] - phase_start[i]
            phase_team[i] = phase_team[i-1]
            phase_start_x[i] = phase_start_x[i-1]
            phase_start_y[i] = phase_start_y[i-1]
            phase_start_type_name[i] = phase_start_type_name[i-1]
            time_middle_third_start[i] = time_middle_third_start[i-1]

        #make box indicators work for defending team
        if (team_id[i] != phase_team[i]) and (start_own_box_ind[i]):
            start_box_ind[i] = True
        if (team_id[i] != phase_team[i]) and (end_own_box_ind[i]):
            end_box_ind[i] = True


        #set whether ball has entered box, prog action has occurred, or ball has left last 25% of pitch in current phase
        entered_box_phase[i]=np.any(start_box_ind[phase_id == phase_id[i]] | end_box_ind[phase_id == phase_id[i]])
        prog_action_phase[i]=np.any(prog_action[phase_id == phase_id[i]])
        left_last_quarter_phase=np.any(x_end[phase_id == phase_id[i]] < (106/4)*3)
        phase_total_distance[i] = np.sqrt((x_start[i] - phase_start_x[i])**2 + (y_start[i] - phase_start_y[i])**2)


        # Increment pass and possaction counters
        # if type_name[i] in ['pass', 'cross']:
        #     num_passes_phase += 1
        if possaction[i]:
            num_possaction_phase += 1
        # Time in middle third
        if (x_start[i] >= 35) and (time_middle_third_start[i]==None):
            time_middle_third_start[i] = time_seconds[i]
        elif (x_start[i] < 35):
            time_middle_third_start[i] = None

        if kickoff[i]:
            phase[i] = 'kickoff'
            force_new_phase[i] = True

        elif (type_name[i] == 'throw_in') and (x_start[i] >= ((106 / 4) * 3)):
            phase[i] = 'attacking_throw_in'
            force_new_phase[i] = True

        elif (type_name[i] == 'throw_in') and (x_start[i] < ((106 / 4) * 3)):
            # Look ahead to see if ball enters the box or 25m arc within 5 seconds
            j = i + 1
            while j < len(data) and (time_seconds[j] - time_seconds[i] <= 5) and (period_id[j] == period_id[i]):
                # Check if either condition is met in the lookahead window.
                
                #check for fk team action in arc or in box
                if (start_box_ind[j] or (start_goal_dist[j] <= 25)) and (team_id[j] == team_id[i]) and (home[i] == 'home'):
                    phase[i] = 'attacking_throw_in'
                    future_attacking_throw_in_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                #check for defending team action in arc or in box if defending team is away
                if (start_own_box_ind[j] or (velocity_5s_homegoaldist[j] <= 25)) and (team_id[j] != team_id[i]) and (not(possaction[j])) and (home[i] == 'home'):
                    phase[i] = 'attacking_throw_in'
                    future_attacking_throw_in_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                #check for defending team action in arc or in box if defending team is home
                if (start_own_box_ind[j] or (velocity_5s_awaygoaldist[j] <= 25)) and (team_id[j] != team_id[i]) and (not(possaction[j])) and (home[i] == 'away'):
                    phase[i] = 'attacking_throw_in'
                    future_attacking_throw_in_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                j += 1
            else:
                # If the loop finishes without a break, then no condition was met.
                phase[i] = 'possession_throw_in'
            force_new_phase[i] = True


        elif type_name[i] in ['corner_short', 'corner_crossed']:
            phase[i] = 'corner'
            force_new_phase[i] = True

        elif type_name[i] == 'shot_freekick':
            phase[i] = 'attacking_freekick'
            force_new_phase[i] = True

        elif (type_name[i] in ['freekick_short', 'freekick_crossed']) and ((start_goal_dist[i] < 35) or (end_goal_dist[i] < 35)):
            phase[i] = 'attacking_freekick'
            force_new_phase[i] = True

        elif (type_name[i] in ['freekick_short', 'freekick_crossed']):
            # Look ahead to see if ball enters the box or 25m arc within 5 seconds
            j = i + 1
            while j < len(data) and (time_seconds[j] - time_seconds[i] <= 5) and (period_id[j] == period_id[i]):
                # Check if either condition is met in the lookahead window.
                
                #check for fk team action in arc or in box
                if (start_box_ind[j] or (start_goal_dist[j] <= 25)) and (team_id[j] == team_id[i]) and (home[i] == 'home'):
                    phase[i] = 'attacking_freekick'
                    future_attacking_freekick_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                #check for defending team action in arc or in box if defending team is away
                if (start_own_box_ind[j] or (velocity_5s_homegoaldist[j] <= 25)) and (team_id[j] != team_id[i]) and (not(possaction[j])) and (home[i] == 'home'):
                    phase[i] = 'attacking_freekick'
                    future_attacking_freekick_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                #check for defending team action in arc or in box if defending team is home
                if (start_own_box_ind[j] or (velocity_5s_awaygoaldist[j] <= 25)) and (team_id[j] != team_id[i]) and (not(possaction[j])) and (home[i] == 'away'):
                    phase[i] = 'attacking_freekick'
                    future_attacking_freekick_trigger=True
                    break  # Exit the loop immediately when a condition is met.
                j += 1
            else:
                # If the loop finishes without a break, then no condition was met.
                phase[i] = 'possession_freekick'
            force_new_phase[i] = True

        elif future_attacking_freekick_trigger:
            phase[i] = 'attacking_freekick'
            #if we've hit the event that converts a possession fk to attacking, allow phase to end from normal triggers
            #check for fk team action that would convert phase
            if(start_box_ind[i] or start_goal_dist[i] <= 25) and (team_id[i]==phase_team[i]):
                future_attacking_freekick_trigger=False
            #check for defending team action that would convert phase if defending team is away
            elif(start_own_box_ind[i] or (velocity_5s_homegoaldist[i] <= 25)) and (team_id[i] != phase_team[i]) and (not(possaction[i])) and (home[i] == 'away'):
                future_attacking_freekick_trigger=False
            #check for defending team action that would convert phase if defending team is home
            elif(start_own_box_ind[i] or (velocity_5s_awaygoaldist[i] <= 25)) and (team_id[i] != phase_team[i]) and (not(possaction[i])) and (home[i] == 'home'):
                future_attacking_freekick_trigger=False
        
        elif future_attacking_throw_in_trigger:
            phase[i] = 'attacking_throw_in'
            #if we've hit the event that converts a possession throw in to attacking, allow phase to end from normal triggers
            #check for fk team action that would convert phase
            if(start_box_ind[i] or start_goal_dist[i] <= 25) and (team_id[i]==phase_team[i]):
                future_attacking_throw_in_trigger=False
            #check for defending team action that would convert phase if defending team is away
            elif(start_own_box_ind[i] or (velocity_5s_homegoaldist[i] <= 25)) and (team_id[i] != phase_team[i]) and (not(possaction[i])) and (home[i] == 'away'):
                future_attacking_throw_in_trigger=False
            #check for defending team action that would convert phase if defending team is home
            elif(start_own_box_ind[i] or (velocity_5s_awaygoaldist[i] <= 25)) and (team_id[i] != phase_team[i]) and (not(possaction[i])) and (home[i] == 'home'):
                future_attacking_throw_in_trigger=False
    
        elif (type_name[i] == 'goalkick') and (x_end[i] >= 42):
            phase[i] = 'long_goalkick'
            force_new_phase[i] = True
        elif (type_name[i] == 'goalkick') and (x_end[i] < 42):
            phase[i] = 'short_goalkick'
            force_new_phase[i] = True

        elif type_name[i] == 'shot_penalty':
            phase[i] = 'penalty'
            force_new_phase[i] = True
        
        elif (type_name[i] in ['freekick_short', 'freekick_crossed']) and (indirect_free_kick_zone[i]):
            phase[i]='indirect_freekick'
            force_new_phase[i] = True
        
        elif (phase[i]=='attacking_freekick') and ((start_goal_dist[i] > 35) or (phase_time[i] > 12)) and (team_id[i] == phase_team[i]):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (type_name[i] in ['keeper_save', 'keeper_claim', 'keeper_pick_up', 'keeper_catch', 'keeper_punch']):
            phase[i] = phase[i]
        
        elif (phase[i] == 'attacking_freekick') and (loose_ball_action[i] and loose_ball_action[i-1]) and start_box_ind[i]:
            phase[i] = 'attacking_freekick_second_phase'
        
        elif (phase[i] == 'attacking_freekick') and (type_name[i-1] in ['clearance', 'keeper_punch','keeper_pick_up','keeper_claim','keeper_save']) and (not (possaction[i]) and (team_id[i] != phase_team[i])):
            phase[i] = 'attacking_freekick_second_phase'
        
        elif (phase[i] == 'attacking_freekick') and (entered_box_phase[i]) and (not start_box_ind[i]) and (team_id[i] == phase_team[i]):
            phase[i] = 'attacking_freekick_second_phase'
        
        elif (phase[i] == 'corner') and (phase_start_type_name[i] == 'corner_short') and (phase_time[i] > 6) and (not entered_box_phase[i]) and (team_id[i]==phase_team[i]):
            phase[i] = 'corner_second_phase'
        
        elif (phase[i] == 'corner') and (loose_ball_action[i] and loose_ball_action[i-1]) and start_box_ind[i]:
            phase[i] = 'corner_second_phase'

        elif (phase[i] == 'corner') and (not start_box_ind[i]) and entered_box_phase[i] and (phase_start_type_name[i] == 'corner_long') and (team_id[i] == phase_team[i]):
            phase[i] = 'corner_second_phase'
        
        elif (phase[i]=='corner') and (phase_time[i] > 10) and (not left_last_quarter_phase) and (team_id[i] == phase_team[i]):
            phase[i] = 'corner_second_phase'
        
        elif (phase[i]=='corner') and (x_start[i] < ((106/3)*2)) and (team_id[i] == phase_team[i]):
            if x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        elif (time_seconds[i] - time_seconds[i-1] >= 20):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
            force_new_phase[i] = True

        elif (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                               'attacking_freekick',
                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
            phase[i] = 'high_ball'
        
        elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                               'attacking_freekick',
                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
            phase[i] = 'high_ball'

        elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                               'attacking_freekick',
                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
            phase[i] = 'high_ball'

        elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                   'attacking_freekick',
                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                   'high_ball']):
            phase[i] = 'loose_ball'
        
        elif type_name[i-1] in ['keeper_catch','keeper_pick_up','keeper_claim'] and (position[i]=='GK') and ( (type_name[i]=='dribble' and ((time_seconds[i]-time_seconds[i-1])<=1.5)) or (type_name[i]=='pass' and ((time_seconds[i]-time_seconds[i-1])<=3))) and (prog_action[i]):
            phase[i] = 'counterattack'
            force_new_phase[i] = True
        
        elif type_name[i-1] in ['keeper_catch','keeper_pick_up','keeper_claim'] and (team_id[i] != phase_team[i]) and possaction[i]:
            phase[i] = 'buildup'
            force_new_phase[i] = True
        
        elif defaction[i-1] and possaction[i] and (team_id[i]!=phase_team[i]) and (start_goal_dist[i]<=35) and (phase[i] != 'loose_ball') and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
            phase[i] = 'high_transition'
            force_new_phase[i] = True
        
        elif defaction[i-1] and possaction[i] and (team_id[i]!=phase_team[i]) and (velocity_5s[i] >= 6.5) and (velocity_5s_prog_perc[i] >= .075) and (velocity_5s_chg[i] >= 25) and (velocity_5s_prog_perc_chg[i] >= .25) and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
            phase[i] = 'counterattack'
            force_new_phase[i] = True   
        
        elif defaction[i-1] and possaction[i] and (team_id[i]!=phase_team[i]) and (phase[i] != 'loose_ball') and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
            phase[i] = 'securing_possession'
            force_new_phase[i] = True
        
        #below is for if a team wins the ball back but more than 6 seconds passes between events - no longer a transition
        elif defaction[i-1] and possaction[i] and (team_id[i]!=phase_team[i]) and (phase[i] != 'loose_ball'):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
            force_new_phase[i] = True
        
        elif (phase[i] == 'securing_possession') and (velocity_5s[i] >= 6.5) and (velocity_5s_prog_perc[i] >= .075) and (velocity_5s_chg[i] >= 25) and (velocity_5s_prog_perc_chg[i] >= .25) and (team_id[i]==phase_team[i]):
            phase[i] = 'counterattack'
            securing_to_other_transition[i] = True
            # # Reset the entire phase as counter now that the attack has moved quickly enough
            # phase[phase_id == phase_id[i]] = 'counterattack'

        elif (phase[i] == 'kickoff') and (phase_time[i]>6.0) and (team_id[i] == phase_team[i]) and possaction[i]:
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i] == 'possession_throw_in') and (phase_total_distance[i] > 21) and (team_id[i] == phase_team[i]) and possaction[i]:
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif(phase[i] in ['buildup','progression','short_goalkick','possession_freekick']) and (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (velocity_5s_chg[i] >= 25) and (velocity_5s_prog_perc_chg[i] >= .25) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
            phase[i] = 'accelerated_possession'

        elif (phase[i] == 'possession_freekick')  and (team_id[i] == phase_team[i]) and possaction[i]:
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        elif (phase[i] in ['long_goalkick', 'short_goalkick']) and (phase_time[i] > 6) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (team_id[i] == phase_team[i]) and (not(defaction[i])) and (phase[i] == 'short_goalkick'):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i] =='short_goalkick') and (x_start[i] > (106/3)) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (team_id[i] == phase_team[i]) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (
                (phase[i] == 'attacking_throw_in') and (
                    ((not entered_box_phase[i]) and (phase_time[i] > 6)) or
                    ((phase_time[i] > 12 and (not start_box_ind[i])) or (x_start[i] < (106/3)*2))
                ) and (team_id[i] == phase_team[i])
            ):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='attacking_freekick_second_phase') and ((start_goal_dist[i] > 35) or (phase_time[i] > 12)) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        elif (phase[i]=='corner_second_phase') and ((x_start[i] < (106/4)*3) or (phase_time[i] > 12)) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        elif (phase[i] == 'penalty') and ((not start_box_ind[i])) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='indirect_freekick') and (not start_box_ind[i]) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='loose_ball') and (distance_prev[i-1] > 20) and (prog_perc_prev[i-1] <= -.1) and possaction[i]:
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <35:
                phase[i] = 'high_transition'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='loose_ball') and ((type_name[i-1]=='foul') or (type_name[i-2]=='foul')):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <35:
                phase[i] = 'high_transition'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='loose_ball') and (possaction[i-1]) and (not (pass_from_onetouch_interception[i-1])) and (possaction[i]):

            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (velocity_5s[i] > 6.5) and (velocity_5s_prog_perc[i] > .075) and (not(defaction[i])):
                phase[i] = 'accelerated_possession'
            elif start_goal_dist[i] <35:
                phase[i] = 'high_transition'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
            
        elif (phase[i]=='high_ball') and (not high_ball_action[i-1]) and (not high_ball_action[i]):
            if ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1]))and (not ((distance_prev[i]>20) and (prog_dist_prev[i]<=-.1))) and (not phase[i] in ['attacking_throw_in',
                                                                               'attacking_freekick',
                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'loose_ball'
            if (defaction[i]):
                phase[i] = 'loose_ball'
            elif defaction[i-1] and possaction[i] and (velocity_5s[i] >= 6.5) and (velocity_5s_prog_perc[i] >= .075) and (velocity_5s_chg[i] >= 25) and (velocity_5s_prog_perc_chg[i] >= .25) and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
                phase[i] = 'counterattack'                                                                                                                                             
            elif defaction[i-1] and possaction[i] and (team_id[i]!=team_id[i-1]) and (start_goal_dist[i]<=25) and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
                phase[i] = 'high_transition'
            elif defaction[i-1] and possaction[i] and (team_id[i]!=team_id[i-1]) and (start_goal_dist[i]<=35) and (start_goal_dist[i-1]<=35) and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
                phase[i] = 'high_transition'
            elif defaction[i-1] and possaction[i] and (team_id[i]!=team_id[i-1])  and (((time_seconds[i]-time_seconds[i-1])<=6) or ((time_seconds[i]-time_seconds[i-1])<=3 and (type_name[i]=='dribble'))):
                phase[i] = 'securing_possession'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        #force the first non high ball action in the phase to not follow any other rules
        elif (phase[i]=='high_ball') and (not high_ball_action[i]) and (high_ball_action[i-1]):
            phase[i] = 'high_ball'
        
        elif (phase[i] == 'securing_possession') and (start_goal_dist[i]<=35) and (team_id[i] == phase_team[i]):
            phase[i] = 'high_transition'
            securing_to_other_transition[i] = True
        
        elif (phase[i] == 'securing_possession') and ((possaction[i-1] and (prog_perc[i-1] < -.1) and (distance[i] > 10)) or (phase_time[i] > 5)) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (i < len(data)-1) and (phase[i]=='counterattack') and (
            (((velocity_5s[i] < 6.5) or (velocity_5s_prog_perc[i] < .075)) and 
             ((velocity_5s[i-1] < 6.5) or (velocity_5s_prog_perc[i-1] < .075)) and 
             ((velocity_5s[i+1] < 6.5) or (velocity_5s_prog_perc[i+1] < .075))) and 
             ((phase_id[i] == phase_id[i-1]))  and 
             ((period_id[i] == period_id[i+1]))
             ) and (team_id[i] == phase_team[i])and (team_id[i-1] == phase_team[i-1])and (team_id[i] == team_id[i+1]):
            
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <= 35 and (phase_time[i] <= 10):
                phase[i] = 'high_transition'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='counterattack') and (possaction[i] and (prog_perc[i] < -.1)) and (team_id[i] == phase_team[i]) and (not defaction[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <= 35 and (phase_time[i] <= 10):
                phase[i] = 'high_transition'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='high_transition') and ((start_goal_dist[i] > 35) or (phase_time[i] > 10)) and (team_id[i] == phase_team[i]):
            if (type_name[i]=='pass') and (bodypart[i]=='head') and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                               'attacking_freekick',
                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-1]) and (not pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                                               'attacking_freekick',
                                                                                                                                                               'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif (high_ball_action[i]) and (high_ball_action[i-2]) and (pass_from_onetouch_interception[i]) and (not phase[i].replace('defending_','') in ['attacking_throw_in',
                                                                                                                                   'attacking_freekick',
                                                                                                                                   'corner', 'corner_second_phase','attacking_freekick_second_phase']):
                 phase[i] = 'high_ball'
            elif ((loose_ball_action[i] and loose_ball_action[i-1]) or (loose_ball_action[i] and pass_from_onetouch_interception[i-1])) and (not phase[i] in ['attacking_throw_in',
                                                                                       'attacking_freekick',
                                                                                       'corner', 'corner_second_phase','attacking_freekick_second_phase',
                                                                                       'high_ball']):
                phase[i] = 'loose_ball'
            elif start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'

        elif (prog_perc[i-1] < -.1) and (x_end[i-1] < 106/3)  and (team_id[i] == phase_team[i]) and (not phase[i] in ['kickoff', 'short_goalkick','possession_throw_in']):
            phase[i] = 'buildup'
        
        elif (phase[i]=='buildup') and (x_start[i] > (106/2) and x_end[i] > (106/2)) and possaction[i] and (team_id[i] == phase_team[i]):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif x_start[i] >= 106/2:
                phase[i] = 'progression'
        
        #leverkusen rule
        elif (phase[i]=='buildup') and (x_start[i] > (106/3)) and (x_start[i-1] > (106/3))and (x_start[i-2] > (106/3)) and possaction[i] and possaction[i-1] and possaction[i-2] and (team_id[i] == phase_team[i]) and (team_id[i-1] == phase_team[i]) and (team_id[i-2] == phase_team[i]) and ((time_seconds[i] - time_middle_third_start[i])>=5):
            phase[i] = 'progression'
        
        elif (phase[i]=='progression') and (
            (start_goal_dist[i]<35) and (start_goal_dist[i-1]<35) and 
                (phase_team[i]==phase_team[i-1])) and possaction[i]:
            phase[i] = 'finishing'
            phase_id[i] = phase_id[i]+1
            if 'progression' in phase[i-1]:
                phase[i-1]=phase[i-1].replace('progression', 'finishing')
                phase_id[i-1] = phase_id[i]
                phase_start[i]=time_seconds[i-1]
                phase_start_type_name[i]=type_name[i-1]
                phase_start_x[i]=x_start[i-1]
                phase_start_y[i]=y_start[i-1]
        
        elif (phase[i]=='progression') and (start_goal_dist[i]<25) and possaction[i] and (team_id[i] == phase_team[i]):
            phase[i] = 'finishing'
        
        elif (phase[i]=='progression') and (start_goal_dist[i]<=35) and (end_goal_dist[i]<25) and possaction[i] and (team_id[i] == phase_team[i]):
            phase[i] = 'finishing'

        elif (phase[i]=='finishing') and (x_start[i] <= 106/3) and (x_start[i-1] <= 106/3) and possaction[i] and team_id[i]==team_id[i-1]:
            phase[i] = 'buildup'
        
        elif (phase[i]=='finishing') and (start_goal_dist[i]>35) and (start_goal_dist[i-1]>35) and (phase_id[i]==phase_id[i-1]) and possaction[i] and team_id[i]==team_id[i-1]:
            phase[i] = 'progression'
            phase[i-1]=phase[i-1].replace('finishing', 'progression')
            phase_id[i]=phase_id[i]+1
            phase_id[i-1]=phase_id[i]
            phase_start[i]=time_seconds[i-1]
            phase_start_type_name[i]=type_name[i-1]
            phase_start_x[i]=x_start[i-1]
            phase_start_y[i]=y_start[i-1]
        
        elif (phase[i]=='progression') and (x_start[i] <= 106/3) and (x_start[i-1] <= 106/3)and (phase_id[i]==phase_id[i-1]) and possaction[i] and team_id[i]==team_id[i-1]:
            phase[i] = 'buildup'
            phase[i-1]=phase[i-1].replace('progression', 'buildup')
            phase_id[i]=phase_id[i]+1
            phase_id[i-1]=phase_id[i]
            phase_start[i]=time_seconds[i-1]
            phase_start_type_name[i]=type_name[i-1]
            phase_start_x[i]=x_start[i-1]
            phase_start_y[i]=y_start[i-1]
        
        
        elif (i < len(data)-1) and (phase[i] == 'accelerated_possession') and (
            (((velocity_5s[i] < 6.5) or (velocity_5s_prog_perc[i] < .075)) and 
             ((velocity_5s[i-1] < 6.5) or (velocity_5s_prog_perc[i-1] < .075)) and 
             ((velocity_5s[i+1] < 6.5) or (velocity_5s_prog_perc[i+1] < .075))) and 
             ((phase_id[i] == phase_id[i-1])) and 
             ((period_id[i] == period_id[i+1]))
             ) and (team_id[i] == phase_team[i])and (team_id[i-1] == phase_team[i-1])and (team_id[i] == team_id[i+1]):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif(phase[i] == 'accelerated_possession') and (possaction[i]) and (prog_perc[i] < -.1):
            if start_goal_dist[i] <25:
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <= 35) and (end_goal_dist[i] < 25):
                phase[i] = 'finishing'
            elif (start_goal_dist[i] <=35) and (start_goal_dist[i-1] <=35):
                phase[i] = 'finishing'
            elif x_start[i] >= 106/3:
                phase[i] = 'progression'
            elif x_start[i] < 106/3:
                phase[i] = 'buildup'
        
        elif (phase[i]=='accelerated_possession') and (
            (start_goal_dist[i]<35) and (start_goal_dist[i-1]<35) and 
                (phase_team[i]==phase_team[i-1])) and (phase[i-1]=='progression') and possaction[i]:
            phase[i-1] = phase[i-1].replace('progression', 'finishing')
            #increment previous phase id if it was part of a longer progression phase so it gets its own id
            if phase_id[i-1]==phase_id[i-2]:
                phase[i-1]=phase[i-1]+1
                phase_id[i] = phase_id[i]+1#incrementing this one too so that it gets a new id of its own in the final checks

        #change counters if new phase
        if (phase[i] != phase[i-1].replace('defending_',''))  or (force_new_phase[i]):
            # num_passes_phase=0
            num_possaction_phase=0
            phase_id[i]+=1
            
            phase_start_x[i]=x_start[i]
            phase_start_y[i]=y_start[i]
            phase_total_distance[i] = 0
        #dont reset time since phase start and team for second phases of corners and free kicks or if securing possession moved to high transition
        if ((phase[i] != phase[i-1].replace('defending_','')) and (not phase[i] in ['corner_second_phase','attacking_freekick_second_phase', 'defending_corner_second_phase', 'defending_attacking_freekick_second_phase','corner']) and (phase[i] != ('defending_'+phase[i-1])) and (not securing_to_other_transition[i])) or (force_new_phase[i]):
            phase_start[i]=time_seconds[i]
            phase_start_type_name[i]=type_name[i]
            phase_team[i] = team_id[i]
            phase_time[i]=0

        #rewrite phase start time with defensive action time if phase is a transition
        if ((phase[i] != phase[i-1].replace('defending_',''))  or (force_new_phase[i])) and (phase[i] in ['counterattack', 'securing_possession', 'high_transition']) and (not securing_to_other_transition[i]):
            phase_start[i]=time_seconds[i-1]
            phase_time[i]=time_seconds[i]-phase_start[i]
        #rewrite phase start type name for corners because some non crossed corners should be long corners
        if corner_short[i]:
            phase_start_type_name[i] = 'corner_short'
        elif corner_long[i]:
            phase_start_type_name[i] = 'corner_long'

        #rename to defending if the team is not the team in possession and not in contested
        if team_id[i] != phase_team[i] and (not phase[i] in ['loose_ball', 'high_ball']):
            phase[i] = f"defending_{phase[i]}"
        

    # Assign the new columns back to the DataFrame
    data['phase'] = phase
    data['phase_time'] = phase_time
    data['phase_id'] = phase_id
    data['phase_team'] = phase_team
    data['velocity_5s_vert'] = velocity_5s
    data['velocity_5s_prog_perc'] = velocity_5s_prog_perc

    # Add phase type- any changes to this might affect sequence IDs
    #mark phase_team as None when phase is loose_ball or high_ball - vectorized approach for better performance
    data.loc[data['phase'].isin(['loose_ball', 'high_ball']), 'phase_team'] = None

    #remove defending prefix from phase column
    data['phase_nodef'] = data['phase'].str.replace('defending_', '', regex=False)

    #create phase_type
    data['phase_type'] = 'open_play'
    data.loc[data['phase_nodef'].isin(['loose_ball', 'high_ball']), 'phase_type'] = 'contested'
    data.loc[data['phase_nodef'].isin(['attacking_freekick', 'attacking_freekick_second_phase',
                                         'attacking_throw_in','corner','corner_second_phase','indirect_freekick']), 'phase_type'] = 'set_piece'
    data.loc[data['phase_nodef']=='penalty', 'phase_type'] = 'penalty'
    data = data.drop(columns=['phase_nodef'])

    return data

def add_possession_sequence_ids(data):
    # Add possession_id column 
    # First fill NaN phase_team with a placeholder value to detect consecutive NaN values
    # NaN is for contested phases
    data['phase_team_filled'] = data['phase_team'].fillna(-1)

    # Create masks for game changes and team changes that ignore contested phases
    game_change = data['game_id'] != data['game_id'].shift(1)
    team_change_without_contested = (data['phase_team_filled'] != data['phase_team_filled'].shift(1)) & \
                  (data['phase_team_filled'] != -1)

    # Calculate possession id
    data['possession_change'] = game_change | team_change_without_contested
    data['possession_id'] = data.groupby('game_id')['possession_change'].cumsum()



    # Additional masks for sequence ID if needed
    throw_in_mask = data['type_name']=='throw_in'
    corner_mask = data['type_name'].isin(['corner_short', 'corner_crossed'])
    free_kick_mask = data['type_name'].isin(['freekick_crossed', 'freekick_short','shot_freekick'])
    penalty_mask = data['type_name']=='shot_penalty'

    # New mask for team change that does not ignore contested phases
    team_change_with_contested = (data['phase_team_filled'] != data['phase_team_filled'].shift(1))
    phase_type_change = (data['phase_type'] != data['phase_type'].shift(1))
    data['sequence_change'] = game_change | team_change_with_contested | throw_in_mask | corner_mask | free_kick_mask | penalty_mask | phase_type_change
    data['sequence_id'] = data.groupby(['game_id'])['sequence_change'].cumsum()

    # Clean up intermediate columns
    data = data.drop(columns=['phase_team_filled', 'possession_change','sequence_change'])

    return data

def add_gamestate(data):
    # Step 1: Add opposing team information by reversing home/away to join to events on game_id and home
    teams_home_reversed = data[['game_id', 'team_id', 'home']].drop_duplicates()
    teams_home_reversed = teams_home_reversed.rename(columns={'team_id': 'opposing_team_id'})
    #mark home as away and vice versa
    teams_home_reversed['home'] = np.where(teams_home_reversed['home'] == "home", "away", "home")
    data = data.merge(teams_home_reversed, on=['game_id', 'home'], how='left')
    
    # Step 2: Sort by game_id and period
    data = data.sort_values(by=['game_id', 'action_id']).reset_index(drop=True)
    
    # Step 3: Identify scoring team
    conditions = [
        data['result_name'] == "owngoal",
        data['goal'] == 1,
        
    ]
    choices = [
        data['opposing_team_id'],  # Assign opposing_team_id for own goals
        data['team_id']  # Assign team_id for goals
        
    ]
    data['scoring_team'] = np.select(conditions, choices, default=None)
    data['scoring_team'] = data.groupby(['game_id'])['scoring_team'].fillna(method='ffill').fillna('No Goal')
    
    # Step 4: Calculate scores for each team. At this point own goals will be counted as goals for the team that scored them, so we need to adjust this after
    data['total_score'] = data.groupby('game_id')['goal'].cumsum()
    data['team_score'] = data.groupby(['game_id', 'scoring_team'])['goal'].cumsum()
    data['opposing_score'] = data['total_score'] - data['team_score']
    
    # Step 7: Calculate goals for and against, different from score because they adjust for own goals
    data['goals_for'] = np.where(data['team_id'] == data['scoring_team'], data['team_score'], data['opposing_score'])
    data['goals_against'] = data['total_score'] - data['goals_for']
    
    
    # Step 6: Fill scores downward for all rows
    data['goals_for'] = data.groupby(['game_id','team_id'])['goals_for'].fillna(method='ffill').fillna(0)
    data['total_score'] = data.groupby(['game_id','team_id'])['total_score'].fillna(method='ffill').fillna(0)
    data['goals_against'] = data['total_score'] - data['goals_for']
    data['gd'] = data['goals_for'] - data['goals_against']
    
    # Step 8: Drop unnecessary columns
    data = data.drop(columns=['total_score', 'scoring_team', 'team_score', 'opposing_score'])

    return data