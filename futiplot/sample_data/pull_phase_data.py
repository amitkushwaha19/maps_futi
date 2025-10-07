# %% Cell 1: Imports and Database Connection
import os
import pandas as pd
from sqlalchemy import text
from futiplot import *  # Assumes db_connect() is defined in futiplot

# Connect to the database using your custom connection function.
engine = db_connect()


# %% Cell 2: Pull Data from vaep_python_sb_phases
# Define a query to pull all records from the vaep_python_sb_phases table.
query = text("SELECT * FROM vaep_python_sb_phases")

with engine.connect() as conn:
    phases_df = pd.read_sql_query(query, conn)

# Sort the data by 'action_id' before proceeding
phases_df = phases_df.sort_values("action_id").drop_duplicates()

print("Total records retrieved:", len(phases_df))
print("Preview of data:")
print(phases_df.head())


#%%
phases_df.to_csv("/users/johnmuller/desktop/phases.csv")


# %% Cell 3: Create Folder and Save CSVs per Unique match_id
# Create the directory "Leverkusen" if it doesn't exist.
output_folder = "Leverkusen"
os.makedirs(output_folder, exist_ok=True)

# Group the DataFrame by match_id and save each group as a separate CSV file.
# Change "match_id" to the appropriate column name if it's different.
for match_id, group in phases_df.groupby("match_id"):
    # Construct a filename using the match_id (you can adjust the naming as needed)
    output_file = os.path.join(output_folder, f"game_{match_id}.csv")
    group.to_csv(output_file, index=False)
    print(f"Saved {len(group)} rows to {output_file}")

# %%
