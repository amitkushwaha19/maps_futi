import numpy as np
import pandas as pd

def get_deserved_result(data, n_sims=10000):
    unique_teams = np.unique(data['team_name'])

    # Step 1: Filter for shots
    data = data[data['type_name'].isin(['shot', 'shot_penalty', 'shot_freekick'])].copy()

    # Step 2: Handle case where no shots are found
    if data.empty:
        # Return default result with 33.3% for each team
        results = {
            'draw_percentage': 33.3,
            **{f"team_{team}_win_percentage": 33.3 for team in unique_teams}
        }
        return results

    # Step 3: Compute team names and indices
    #overwrite team names from before now that shots are filtered
    unique_teams, team_indices = np.unique(data['team_name'], return_inverse=True)

    # Step 4: Simulate whether each shot is a goal for each match simulation
    simulated_goals = np.random.rand(len(data), n_sims) < data['xg_team'].values[:, None]

    # Step 5: Sum goals for each team in each simulation
    team_goals = np.zeros((n_sims, len(unique_teams)), dtype=int)
    for sim_idx in range(n_sims):
        np.add.at(team_goals[sim_idx], team_indices, simulated_goals[:, sim_idx])

    # Step 6: Determine the winner and draw percentage
    max_goals = team_goals.max(axis=1)  # Maximum goals scored in each simulation
    is_draw = (team_goals == max_goals[:, None]).sum(axis=1) > 1  # Check if it's a draw
    winners = (team_goals == max_goals[:, None]) & ~is_draw[:, None]  # Identify winners

    # Calculate percentages
    draw_percentage = is_draw.mean() * 100
    win_percentages = winners.mean(axis=0) * 100

    # Step 9: Return results
    results = {
        'draw_percentage': draw_percentage,
        **{f"{team}_win_percentage": win_percentages[i] for i, team in enumerate(unique_teams)}
    }
    
    return results