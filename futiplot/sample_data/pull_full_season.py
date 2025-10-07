#%%
# Cell 0: connectivity & table names
from futiplot import db_connect
from sqlalchemy import text,inspect

engine = db_connect()
inspector = inspect(engine)

for table in inspector.get_table_names():
    cols = [col["name"] for col in inspector.get_columns(table)]
    print(f"{table}: {cols}")

#%%
# Cell 1: check table permissions
for table in ["vaep_mls", "teams_mls", "matches_mls"]:
    try:
        with engine.connect() as conn:
            conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
        print(f"✔️  SUCCESS: able to SELECT from `{table}`")
    except Exception as e:
        print(f"❌  ERROR: cannot SELECT from `{table}` -- {e}")

#%%
# Cell 2: pull vaep + team names + opponent names + match_date in one go
import pandas as pd

sql = text("""
  SELECT
    v.*,
    t.team_name       AS team_name,
    o.team_name       AS opponent_name,
    m.match_date
  FROM vaep_mls    AS v
  LEFT JOIN teams_mls   AS t ON v.team_id     = t.team_id
  LEFT JOIN teams_mls   AS o ON v.opposing_team_id = o.team_id
  LEFT JOIN matches_mls AS m ON v.match_id     = m.match_id
""")

df = pd.read_sql_query(sql, engine)
df.head()

#%%
# Cell 3: (optional) save to CSV
import os
out = os.path.expanduser("~/documents/github/vaep_full.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")

# %%
