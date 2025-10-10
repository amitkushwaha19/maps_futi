import sys
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join('..', 'functions')))
from db_connect import db_connect
from deserved_result_functions import get_deserved_result

load_dotenv()



# --- New Cell ---

#identify new matches and retrieve raw data
con = db_connect('futi')

if con:
    # Get sample of vaep data
    sample_game = 4000
    query = "SELECT game_id, team_id, xg_team, type_name FROM vaep_mls WHERE game_id = %s"
    df = pd.read_sql(query, con, params=[sample_game])

    # get team names and ids
    query = "SELECT team_name, team_id FROM teams_mls"
    teams = pd.read_sql(query, con)

    #join team names to data
    df = df.merge(teams, on='team_id', how='left')

    con.close()

# --- New Cell ---

# Get the deserved result for the sample match
deserved_result = get_deserved_result(df, n_sims=10000) #might need to reduce number of simulations for deployment depending on speed
# Print the deserved result
deserved_result