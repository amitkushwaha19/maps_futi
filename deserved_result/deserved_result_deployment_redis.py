#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import redis
import json

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from deserved_result_functions import get_deserved_result

load_dotenv()

def load_match_from_redis(match_id):
    """Load match data from Redis stream"""
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Read from Redis stream
    stream_key = f"stream:match:{match_id}:events"
    events = r.xrange(stream_key)
    
    if not events:
        return pd.DataFrame()
    
    # Convert to DataFrame
    data_list = []
    for event_id, event_data in events:
        record = {}
        for k, v in event_data.items():
            try:
                record[k] = float(v)
            except (ValueError, TypeError):
                record[k] = v
        data_list.append(record)
    
    df = pd.DataFrame(data_list)
    
    # Add team names
    if 'match_home_team_name' in df.columns and 'match_away_team_name' in df.columns:
        home_team_id = df[df['home'] == 'home']['team_id'].iloc[0] if len(df[df['home'] == 'home']) > 0 else None
        away_team_id = df[df['home'] == 'away']['team_id'].iloc[0] if len(df[df['home'] == 'away']) > 0 else None
        
        df['team_name'] = df['team_id'].map({
            home_team_id: df['match_home_team_name'].iloc[0],
            away_team_id: df['match_away_team_name'].iloc[0]
        })
    
    # Ensure xg_team column is numeric (it should already exist in Redis data)
    if 'xg_team' in df.columns:
        df['xg_team'] = pd.to_numeric(df['xg_team'], errors='coerce')
    else:
        # Fallback: use xg column if xg_team doesn't exist
        df['xg_team'] = pd.to_numeric(df['xg'], errors='coerce')
    
    # Fill NaN values with 0
    df['xg_team'] = df['xg_team'].fillna(0)
    
    return df

# Connect to Redis and get all matches
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
keys = r.keys("stream:match:*:events")
match_ids = [key.split(":")[2] for key in keys]

print(f"Found {len(match_ids)} matches in Redis")

# Process each match
for match_id in match_ids:
    print(f"\nProcessing match {match_id}...")
    
    # Load match data from Redis
    df = load_match_from_redis(match_id)
    
    if df.empty:
        print(f"No data for match {match_id}")
        continue
    
    # Get the deserved result
    deserved_result = get_deserved_result(df, n_sims=10000)
    
    # Save to JSON
    with open(f"deserved_result_{match_id}.json", 'w') as f:
        json.dump(deserved_result, f, indent=2)
    
    # Print result
    print(f"Deserved result for match {match_id}:")
    for key, value in deserved_result.items():
        print(f"  {key}: {value:.2f}%")

print("\nDone!")