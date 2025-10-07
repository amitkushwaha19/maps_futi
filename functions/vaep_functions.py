import pandas as pd
import numpy as np
from scipy import sparse

from functions import spadl_processing_functions

def prep_X(df, feat_names, convert_to_sparse=True):
    """ Preprocess the feature DataFrame:
    - Ensure all feat_names columns are present (new ones get 0)
    - Drop any columns not in feat_names
    - Reorder columns to match feat_names
    - Convert boolean columns to 0/1
    - Fill missing values with 0
    - Replace infinite values with 0
    """
    X = df.reindex(columns=feat_names, fill_value=0).copy()
    # convert boolean to 0/1
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)
    # fill missing with 0
    X.fillna(0, inplace=True)
    #replace infinites with 0
    X.replace([np.inf, -np.inf], 0, inplace=True)
    if convert_to_sparse:
        X = sparse.csr_matrix(X.values)
    return X

def vaep_scores_label(actions, nr_actions=10):
    # Merging goals, owngoals, and team_ids
    actions['owngoal'] = np.where(actions['result_name'] == 'owngoal', 1, 0)
    goals = actions['goal'] == 1
    owngoals = actions['owngoal'] == 1
    
    y = pd.DataFrame({'goal': goals, 'owngoal': owngoals, 'team_id': actions['team_id']})
    
    # Adding future results
    for i in range(1, nr_actions):
        for c in ['team_id', 'goal', 'owngoal']:
            shifted = y[c].shift(-i)
            shifted.iloc[-i:] = y[c].iloc[-1]
            y[f'{c}+{i}'] = shifted
    
    # Calculating scores
    res = y['goal']
    for i in range(1, nr_actions):
        gi = y[f'goal+{i}'] & (y[f'team_id+{i}'] == y['team_id'])
        ogi = y[f'owngoal+{i}'] & (y[f'team_id+{i}'] != y['team_id'])
        res = res | gi | ogi
    
    return pd.DataFrame({'scores': res})

def vaep_concedes_label(actions, nr_actions=10):
    actions['owngoal'] = np.where(actions['result_name'] == 'owngoal', 1, 0)
    goals = actions['goal'] == 1
    owngoals = actions['owngoal'] == 1
    
    y = pd.DataFrame({'goal': goals, 'owngoal': owngoals, 'team_id': actions['team_id']})
    
    # Adding future results
    for i in range(1, nr_actions):
        for c in ['team_id', 'goal', 'owngoal']:
            shifted = y[c].shift(-i)
            shifted.iloc[-i:] = y[c].iloc[-1]
            y[f'{c}+{i}'] = shifted
    
    # Compute the results
    res = y['owngoal']
    for i in range(1, nr_actions):
        gi = y[f'goal+{i}'] & (y[f'team_id+{i}'] != y['team_id'])
        ogi = y[f'owngoal+{i}'] & (y[f'team_id+{i}'] == y['team_id'])
        res = res | gi | ogi
    
    return pd.DataFrame({'concedes': res})

def vaep_game_labels(game, actions):
    tmp = actions[actions['game_id'] == game].copy()
    
    scores = vaep_scores_label(tmp)
    concedes = vaep_concedes_label(tmp)
    
    result = pd.concat([scores, concedes], axis=1)
    result['game_id'] = game
    
    return result

def vaep_add_gamestates(actions, nb_prev_actions=3):
    # Check if nb_prev_actions is at least 1
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    
    actions=actions.reset_index(drop=True)
    states = [actions]  # Initialize list of states with the original actions
    if nb_prev_actions > 1:
        for i in range(1, nb_prev_actions):
            prev_actions = actions.copy()
            prev_actions = prev_actions.groupby(['game_id', 'period_id']).apply(
                lambda group: group.shift(i)
            ).reset_index(drop=True)
            
            states.append(prev_actions)  # Append previous actions to the states list
    
    return states  # Return the list of game states
    # Each element in the list is a full DataFrame of events, one that is normal, then one for each action backwards

def vaep_actiontype_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        
        if i == 0:
            out = pd.DataFrame({
                f'type_id_a{i}': gamestate['type_id']
            })
        else:
            out[f'type_id_a{i}'] = gamestate['type_id']
        
        # Replace NAs with -99
        out = out.fillna(-99)
    
    return out

def vaep_actiontype_onehot_features(gamestates):
    out = pd.DataFrame()
    for i, gamestate in enumerate(gamestates):
        gamestate.reset_index(drop=True, inplace=True)
        for type_id, type_name in spadl_processing_functions.spadl_action_df[['type_id', 'type_name']].values:
            col = "type_" + type_name + "_a" + str(i)
            out[col] = gamestate["type_id"] == type_id
    return(out.astype(int))

def vaep_bodypart_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        # Replace bodypart id for left or right foot with just foot
        gamestate['bodypart_id'] = gamestate['bodypart_id'].replace({4: 0, 5: 0})
        
        if i == 0:
            out = pd.DataFrame({
                f'bodypart_id_a{i}': gamestate['bodypart_id']
            })
        else:
            out[f'bodypart_id_a{i}'] = gamestate['bodypart_id']
        
        # Replace NAs with -99
        out = out.fillna(-99)
    
    return out

def vaep_bodypart_onehot_features(gamestates):
    out = pd.DataFrame()
    for i, gamestate in enumerate(gamestates):
        gamestate.reset_index(drop=True, inplace=True)

        # Replace bodypart id for left or right foot with just foot
        gamestate['bodypart_id'] = gamestate['bodypart_id'].replace({4: 0, 5: 0})

        for bodypart_id, bodypart_name in spadl_processing_functions.spadl_bodypart_df[['bodypart_id','bodypart_name']].values:
            if bodypart_name in ("foot_left", "foot_right"):
                continue
            col = "bodypart_" + bodypart_name + "_a" + str(i)
            out[col] = gamestate["bodypart_id"] == bodypart_id
    return(out.astype(int))

def vaep_result_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        if i == 0:
            out = pd.DataFrame({
                f'result_id_a{i}': gamestate['result_id']
            })
        else:
            out[f'result_id_a{i}'] = gamestate['result_id']
    
    return out

def vaep_result_onehot_features(gamestates):
    out = pd.DataFrame()
    for i, gamestate in enumerate(gamestates):
        gamestate.reset_index(drop=True, inplace=True)

        for result_id, result_name in spadl_processing_functions.spadl_result_df[['result_id','result_name']].values:
            col = "result_" + result_name + "_a" + str(i)
            out[col] = gamestate["result_id"] == result_id
    return(out.astype(int))

def vaep_goalscore_features(gamestates):
    # Arbitrarily assign teamA
    teamA = gamestates[0]['team_id'].iloc[0]
    
    # Only the dataframe for the actual action, not previous actions
    df = gamestates[0].copy()
    
    df['owngoals'] = df['result_name'] == 'owngoal'
    df['teamisA'] = df['team_id'] == teamA
    df['teamisB'] = ~df['teamisA']
    df['goalsteamA'] = (df['goal'] & df['teamisA']) | (df['owngoals'] & df['teamisB'])
    df['goalsteamB'] = (df['goal'] & df['teamisB']) | (df['owngoals'] & df['teamisA'])
    
    df['goalscoreteamA'] = df['goalsteamA'].cumsum() - df['goalsteamA'].astype(int)
    df['goalscoreteamB'] = df['goalsteamB'].cumsum() - df['goalsteamB'].astype(int)
    
    df['goalscore_team'] = np.where(df['teamisA'], df['goalscoreteamA'], df['goalscoreteamB'])
    df['goalscore_opponent'] = np.where(df['teamisA'], df['goalscoreteamB'], df['goalscoreteamA'])
    df['goalscore_diff'] = df['goalscore_team'] - df['goalscore_opponent']
    
    return df[['goalscore_team', 'goalscore_opponent', 'goalscore_diff']]

def vaep_startlocation_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        if i == 0:
            out = pd.DataFrame({
                f'x_start_a{i}': gamestate['x_start'],
                f'y_start_a{i}': gamestate['y_start']
            })
        else:
            out[f'x_start{i}'] = gamestate['x_start']
            out[f'y_start{i}'] = gamestate['y_start']
    
    return out

def vaep_endlocation_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        if i == 0:
            out = pd.DataFrame({
                f'x_end_a{i}': gamestate['x_end'],
                f'y_end_a{i}': gamestate['y_end']
            })
        else:
            out[f'x_end_a{i}'] = gamestate['x_end']
            out[f'y_end_a{i}'] = gamestate['y_end']
    
    return out

def vaep_movement_features(gamestates):

    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        gamestate=gamestate.assign(
            dx=gamestate['x_end'] - gamestate['x_start'],
            dy=gamestate['y_end'] - gamestate['y_start'],
            movement=np.sqrt((gamestate['x_end'] - gamestate['x_start'])**2 + (gamestate['y_end'] - gamestate['y_start'])**2)
        )

        if i == 0:
            out = pd.DataFrame({
                f'dx_a{i}': gamestate['dx'],
                f'dy_a{i}': gamestate['dy'],
                f'movement_a{i}': gamestate['movement']
            })
        else:
            out[f'dx_a{i}'] = gamestate['dx']
            out[f'dy_a{i}'] = gamestate['dy']
            out[f'movement_a{i}'] = gamestate['movement']
    
    return out

def vaep_space_delta_features(gamestates):
    currstate = gamestates[0].reset_index(drop=True)
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates[1:]):
        gamestate=gamestate.assign(
            mov=np.sqrt((gamestate['x_start'] - currstate['x_start'])**2 + (gamestate['y_start'] - currstate['y_start'])**2)
        ).reset_index(drop=True)


        if i == 0:
            out = pd.DataFrame({
                f'mov_a{i+1}': gamestate['mov']
            })
        else:
            out[f'mov_a{i+1}'] = gamestate['mov']
    
    return out

def vaep_startpolar_features(gamestates, goal_x=105, goal_y=68/2):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        gamestate=gamestate.assign(
            dx_goal=abs(goal_x - gamestate['x_start']),
            dy_goal=abs(goal_y - gamestate['y_start']))
        gamestate=gamestate.assign(
            start_dist_to_goal=np.sqrt((goal_x - gamestate['x_start'])**2 + (goal_y - gamestate['y_start'])**2),
            start_angle_to_goal=np.where(gamestate['x_start'] != 0, np.arctan(gamestate['dy_goal'] / gamestate['dx_goal']), np.nan)
        )

        if i == 0:
            out = pd.DataFrame({
                f'start_dist_to_goal_a{i}': gamestate['start_dist_to_goal'],
                f'start_angle_to_goal_a{i}': gamestate['start_angle_to_goal']
            })
        else:
            out[f'start_dist_to_goal_a{i}'] = gamestate['start_dist_to_goal']
            out[f'start_angle_to_goal_a{i}'] = gamestate['start_angle_to_goal']
    
    return out

def vaep_endpolar_features(gamestates, goal_x=105, goal_y=68/2):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        gamestate=gamestate.assign(
            dx_goal=abs(goal_x - gamestate['x_end']),
            dy_goal=abs(goal_y - gamestate['y_end'])    
        )
        gamestate=gamestate.assign(
            end_dist_to_goal=np.sqrt((goal_x - gamestate['x_end'])**2 + (goal_y - gamestate['y_end'])**2),
            end_angle_to_goal=np.where(gamestate['x_end'] != 0, np.arctan(gamestate['dy_goal'] / gamestate['dx_goal']), np.nan)
        )

        if i == 0:
            out = pd.DataFrame({
                f'end_dist_to_goal_a{i}': gamestate['end_dist_to_goal'],
                f'end_angle_to_goal_a{i}': gamestate['end_angle_to_goal']
            })
        else:
            out[f'end_dist_to_goal_a{i}'] = gamestate['end_dist_to_goal']
            out[f'end_angle_to_goal_a{i}'] = gamestate['end_angle_to_goal']
    
    return out

def vaep_team_features(gamestates):
    # These features indicate whether the team was in possession X actions ago
    a0 = gamestates[0].reset_index(drop=True)   # Access the first gamestate
    teamdf = pd.DataFrame(index=range(len(a0)))  # Initialize a new data frame
    
    for i, gamestate in enumerate(gamestates[1:]):  # Iterate over subsequent gamestates
        gamestate=gamestate.reset_index(drop=True)
        teamdf[f'team_{i}'] = gamestate['team_id'] == a0['team_id']  # Compare team IDs
    
    teamdf = teamdf.fillna(False)  # Replace NAs with False
    return teamdf

def vaep_time_features(gamestates):
    
    def calc_time_overall(df):
        period_times = df.groupby(['game_id', 'period_id']).agg(
            period_end_time=('time_seconds', 'max'),
            period_start_time=('time_seconds', 'min')
        ).reset_index()
        
        period_times['period_length'] = period_times['period_end_time'] - period_times['period_start_time']
        period_times['prev_period_length'] = period_times.groupby('game_id')['period_length'].shift(1).fillna(0)
        period_times['total_time_period_start'] = period_times.groupby('game_id')['prev_period_length'].cumsum()
        
        df = df.merge(period_times[['game_id', 'period_id', 'total_time_period_start']], on=['game_id', 'period_id'])
        df['time_seconds_overall'] = df['time_seconds'] + df['total_time_period_start']
        return df[['period_id', 'time_seconds', 'time_seconds_overall']]
    
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        gamestate=calc_time_overall(gamestate)

        if i == 0:
            out = pd.DataFrame({
                f'period_id_a{i}': gamestate['period_id'],
                f'time_seconds_a{i}': gamestate['time_seconds'],
                f'time_seconds_overall_a{i}': gamestate['time_seconds_overall']
            })
        else:
            out[f'period_id_a{i}'] = gamestate['period_id']
            out[f'time_seconds_a{i}'] = gamestate['time_seconds']
            out[f'time_seconds_overall_a{i}'] = gamestate['time_seconds_overall']
    
    return out

def vaep_time_delta_features(gamestates):
    # Time between current action and the time X prev action occurred
    currtime = gamestates[0]['time_seconds'].reset_index(drop=True)

    for i, gamestate in enumerate(gamestates[1:]):
        gamestate=gamestate.assign(
                timea0=currtime,
                time_delta=currtime - gamestate['time_seconds']
            )

        if i == 0:
            out = pd.DataFrame({
                f'time_delta_a{i+1}': gamestate['time_delta']
            })
        else:
            out[f'time_delta_a{i+1}'] = gamestate['time_delta']
    
    return out

def vaep_velocity_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        
        if i == 0:
            out = pd.DataFrame({
                f'velocity_5s_vert_a{i}': gamestate['velocity_5s_vert'],
                f'velocity_5s_prog_perc_a{i}': gamestate['velocity_5s_prog_perc']
            })
        else:
            out[f'velocity_5s_vert_a{i}'] = gamestate['velocity_5s_vert']
            out[f'velocity_5s_prog_perc_a{i}'] = gamestate['velocity_5s_prog_perc']
        
        # Replace NAs with -99
        out = out.fillna(-99)
    
    return out

def vaep_phase_team_features(gamestates):
    # These features indicate whether the team was in possession X actions ago
    a0 = gamestates[0].reset_index(drop=True)   # Access the first gamestate
    out = pd.DataFrame(index=range(len(a0)))  # Initialize a new data frame
    
    for i, gamestate in enumerate(gamestates[1:]):  # Iterate over subsequent gamestates
        gamestate=gamestate.reset_index(drop=True)
        out[f'phase_team_{i}'] = gamestate['phase_team'] == a0['phase_team']  # Compare team IDs
    
    out = out.fillna(False)  # Replace NAs with False
    return out

def vaep_phase_onehot_features(gamestates):
    out = pd.DataFrame()
    for i, gamestate in enumerate(gamestates):
        gamestate.reset_index(drop=True, inplace=True)
        #remove defending_ from phase columns
        gamestate['phase'] = gamestate['phase'].str.replace('defending_', '')
        for phase_name in spadl_processing_functions.spadl_phase_names:
            col = "phase_" + phase_name + "_a" + str(i)
            out[col] = gamestate["phase"] == phase_name
    return(out.astype(int))

def vaep_phase_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        #remove defending_ from phase columns
        gamestate['phase'] = gamestate['phase'].str.replace('defending_', '')
        if i == 0:
            out = pd.DataFrame({
                f'phase_a{i}': gamestate['phase']
            })
        else:
            out[f'phase_a{i}'] = gamestate['phase']
        
        # Replace NAs with kickoff
        #this is kickoff because this only happens in the first action of each period, which is kickoff
        out = out.fillna('kickoff')
    
    return out

def vaep_phase_time_features(gamestates):
    out = pd.DataFrame()
    
    for i, gamestate in enumerate(gamestates):
        
        if i == 0:
            out = pd.DataFrame({
                f'phase_time_a{i}': gamestate['phase_time']
            })
        else:
            out[f'phase_time_a{i}'] = gamestate['phase_time']
        
        # Replace NAs with -99
        out = out.fillna(-99)
    
    return out

def vaep_source_features(gamestates):
    n = len(gamestates[0])
    out = gamestates[0][['source']].copy()
    return out

def vaep_source_onehot_features(gamestates):
    n = len(gamestates[0])
    if gamestates[0]['source'].iloc[1] == 'statsbomb':
        out = pd.DataFrame({
            'source_statsbomb': np.ones(n, dtype=int),
            'source_wyscout': np.zeros(n, dtype=int)
        })
    elif gamestates[0]['source'].iloc[1] == 'wyscout':
        out = pd.DataFrame({
            'source_statsbomb': np.zeros(n, dtype=int),
            'source_wyscout': np.ones(n, dtype=int)
        })
    return out

def vaep_high_pass_features(gamestates):
    features = {}
    for i, gs in enumerate(gamestates):
        # reset index so all series align on row number
        df = gs.reset_index(drop=True)
        # compute high‐pass mask and cast to int
        features[f'high_pass_a{i}'] = (
            (df['type_name'] == 'pass') & (df['height'] == 'high')
        ).astype(int)

    # turn that dict into a DataFrame (columns in order of keys)
    return pd.DataFrame(features)

def vaep_generate_features_game(gamestates):
    feature_functions = [
        # vaep_actiontype_features,
        vaep_actiontype_onehot_features,
        # vaep_bodypart_features,
        vaep_bodypart_onehot_features,
        # vaep_result_features,
        vaep_result_onehot_features,
        # vaep_goalscore_features,
        vaep_startlocation_features,
        vaep_endlocation_features,
        vaep_movement_features,
        vaep_space_delta_features,
        vaep_startpolar_features,
        vaep_endpolar_features,
        vaep_team_features,
        vaep_time_features,
        vaep_time_delta_features,
        # vaep_phase_features,
        vaep_phase_onehot_features,
        vaep_velocity_features,
        vaep_phase_time_features,
        vaep_phase_team_features,
        vaep_source_onehot_features,
        vaep_high_pass_features
    ]
    return pd.concat([func(gamestates) for func in feature_functions], axis=1)

def vaep_generate_features_all_games(df):

    all_features=pd.DataFrame()
    for game in df['game_id'].unique():
        df_game=df[df['game_id']==game]
        gamestates_game = vaep_add_gamestates(df_game, nb_prev_actions=3)
        features_game = vaep_generate_features_game(gamestates_game)
        all_features=pd.concat([all_features, features_game], ignore_index=True)

    return all_features