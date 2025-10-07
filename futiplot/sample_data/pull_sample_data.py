#%%
# from sqlalchemy import text
import pandas as pd
from futiplot import *
from sqlalchemy import inspect

def list_tables_and_columns(engine):
    """
    Lists all tables in the database and their columns.

    Args:
        engine: SQLAlchemy engine object.

    Returns:
        dict: A dictionary where keys are table names and values are lists of column names.
    """
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    table_columns = {}

    for table in tables:
        columns = inspector.get_columns(table)
        column_names = [col['name'] for col in columns]
        table_columns[table] = column_names

    return table_columns

if __name__ == "__main__":
    # Step 1: Establish database connection
    engine = db_connect()

    try:
        # Step 2: Get all tables and their columns
        tables_and_columns = list_tables_and_columns(engine)

        # Step 3: Print the results
        print("Available Tables and Columns:")
        for table, columns in tables_and_columns.items():
            print(f"\nTable: {table}")
            print(f"Columns: {', '.join(columns)}")
    except Exception as e:
        print(f"An error occurred: {e}")

#%%
from sqlalchemy import text
import pandas as pd

def get_match_id(engine, home_team_partial, away_team_partial):
    """
    Fetch the most recent match_id from the matches_mls table where the home team name contains
    the specified substring for the home team, and the away team name contains the
    specified substring for the away team.

    Args:
        engine: SQLAlchemy engine object.
        home_team_partial: Partial name of the home team (e.g., "Brighton").
        away_team_partial: Partial name of the away team (e.g., "Chelsea").

    Returns:
        str: The match_id of the most recent match meeting the criteria, or None.
    """
    query = text("""
        SELECT "match_id"
        FROM matches_mls
        WHERE "match_home_team_name" ILIKE '%' || :home_team_partial || '%'
          AND "match_away_team_name" ILIKE '%' || :away_team_partial || '%'
        ORDER BY "match_date" DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"home_team_partial": home_team_partial, "away_team_partial": away_team_partial}).fetchone()
        return result[0] if result else None  # Access tuple element by index


#%%
def get_game_data(engine, match_id):
    """
    Fetch all data for a specific match_id from the vaep_mls table.

    Args:
        engine: SQLAlchemy engine object.
        match_id: The ID of the game to fetch.

    Returns:
        pandas.DataFrame: DataFrame containing all data for the specified game.
    """
    query = text("""
        SELECT *
        FROM vaep_mls
        WHERE match_id = :match_id
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"match_id": match_id})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df

#%%
if __name__ == "__main__":
    # Step 1: Establish database connection
    engine = db_connect()

    # Step 2: Specify partial team names
    home_team_partial = "Vancouver"
    away_team_partial = "San Diego"

    try:
        # Step 3: Fetch the match_id
        match_id = get_match_id(engine, home_team_partial, away_team_partial)
        if match_id:
            print(f"Most recent match_id for {home_team_partial} (home) vs {away_team_partial} (away): {match_id}")

            # Step 4: Fetch game data using the match_id
            game_data = get_game_data(engine, match_id)
            if not game_data.empty:
                print(f"Data for match_id {match_id} successfully fetched:")
                print(game_data)
            else:
                print(f"No data found for match_id {match_id}.")
        else:
            print(f"No recent game found for {home_team_partial} (home) vs {away_team_partial} (away).")
    except Exception as e:
        print(f"An error occurred: {e}")


# %%
import os

# Define the file path
output_file = os.path.join("/users/johnmuller/documents/github/python_scripts/futiplot/src/futiplot/sample_data/sample_vaep.csv")

# Save the DataFrame to CSV
game_data.to_csv(output_file, index=False)

print(f"Game data saved to {output_file}")

# %%
