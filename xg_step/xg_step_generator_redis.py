#!/usr/bin/env python3
"""
xG Step Generator - Redis Stream Version
========================================

This script generates xG step chart data for matches using Redis streams as data source,
following the same pattern as other Redis generators. It exports frontend-friendly JSON
for easy consumption by the frontend team.

Usage:
    python xg_step_generator_redis.py

Output:
    - JSON files with xG step data for each match
    - Optional figure display (uncomment show_xg_step_figure())
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import redis

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from futiplot.match_card.match_card_charts import df_match_xg_step, compute_timing_meta
    from futiplot.utils import futicolor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class XGStepGeneratorRedis:
    """Generate xG step chart data for a specific match_id using Redis streams"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.xg_step_data = None
        self.original_df = None
        self.meta = None
        self.per_bounds = None
        
    def load_match_data(self):
        """Load match data from Redis streams"""
        print(f"🔍 Loading data for match_id: {self.match_id}")

        try:
            # Normalize match_id to int for stream naming
            match_id = int(self.match_id) if isinstance(self.match_id, str) else self.match_id

            # Connect to Redis and read per-match stream
            r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
            stream_name = f"stream:match:{match_id}:events"

            entries = r.xrange(stream_name, min='-', max='+')

            if not entries:
                print(f"❌ No data found in Redis stream for match_id: {self.match_id} (stream: {stream_name})")
                return False

            # Flatten Redis entries properly
            records = []
            for _entry_id, fields in entries:
                # fields is a dict, decode numeric strings
                record = {}
                for k, v in fields.items():
                    # Convert numeric strings to float if possible
                    try:
                        record[k] = float(v)
                    except (ValueError, TypeError):
                        record[k] = v
                records.append(record)

            df_game = pd.DataFrame(records)

            if df_game.empty:
                print(f"❌ No data found for match_id: {self.match_id}")
                return False

            print(f"✅ Found {len(df_game)} events for match {self.match_id} (from stream)")

            # Ensure required columns exist for xG step
            required_cols = ['type_name', 'x_start', 'y_start', 'team_id', 'xg']
            missing_cols = [col for col in required_cols if col not in df_game.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns for xG step: {missing_cols}")
                return False

            # Convert numeric columns needed for xG step generation FIRST
            numeric_cols = ['x_start', 'y_start', 'team_id', 'player_id', 'xg', 'time_seconds', 'period_id']
            for col in numeric_cols:
                if col in df_game.columns:
                    df_game[col] = pd.to_numeric(df_game[col], errors='coerce')

            # Now filter shot events (after converting xg to numeric)
            shot_events = df_game[
                (df_game.get('xg', 0) > 0) | 
                (df_game['type_name'].astype(str).str.lower() == 'shot')
            ]

            print(f"🏹 Found {len(shot_events)} shot events (before deduplication)")

            # Deduplicate shot events to avoid artificial xG step points
            if not shot_events.empty:
                # Create a unique key for deduplication based on key shot characteristics
                shot_events = shot_events.copy()
                
                # Use a more conservative deduplication approach
                # Only remove events that are EXACTLY identical (likely true duplicates)
                
                # First, try to identify if we have a unique event ID column
                id_cols = ['action_id', 'event_id', 'original_event_id']
                unique_id_col = None
                for col in id_cols:
                    if col in shot_events.columns:
                        unique_id_col = col
                        break
                
                shots_before = len(shot_events)
                
                if unique_id_col:
                    # If we have unique IDs, use those for deduplication (safest)
                    shot_events = shot_events.drop_duplicates(subset=[unique_id_col], keep='first')
                    print(f"🔧 Deduplicated by {unique_id_col}: {shots_before} → {len(shot_events)} (removed {shots_before - len(shot_events)} exact duplicates)")
                else:
                    # Fallback: Only remove EXACT duplicates (no rounding)
                    # This is much safer - only removes truly identical rows
                    exact_dedup_cols = ['team_id', 'time_seconds', 'x_start', 'y_start', 'xg', 'player_id']
                    available_exact_cols = [col for col in exact_dedup_cols if col in shot_events.columns]
                    
                    if available_exact_cols:
                        shot_events = shot_events.drop_duplicates(subset=available_exact_cols, keep='first')
                        print(f"🔧 Deduplicated by exact match: {shots_before} → {len(shot_events)} (removed {shots_before - len(shot_events)} exact duplicates)")
                    else:
                        print(f"⚠️ No suitable columns for deduplication, keeping all {shots_before} shots")
                
                # Update the main dataframe with deduplicated shots
                # Keep all non-shot events + deduplicated shot events
                non_shot_events = df_game[~((df_game.get('xg', 0) > 0) | (df_game['type_name'].astype(str).str.lower() == 'shot'))]
                df_game = pd.concat([non_shot_events, shot_events], ignore_index=True).sort_values('time_seconds', na_position='last')

            print(f"🏹 Final shot events: {len(shot_events)}")

            if shot_events.empty:
                print("⚠️ No shot events found in this match")
                return False

            self.original_df = df_game

            # Compute timing metadata
            try:
                self.meta, self.per_bounds = compute_timing_meta(df_game, bin_seconds=120, plot_gap_seconds=None)
                print(f"⏱️ Computed timing metadata")
            except Exception as e:
                print(f"⚠️ Warning: Could not compute timing metadata: {e}")
                # Create minimal meta data
                self.meta = pd.DataFrame({'D': [6000.0]})  # Default duration
                self.per_bounds = pd.DataFrame()

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def generate_xg_step_data(self):
        """Generate xG step data using the original function"""
        print("📈 Generating xG step data...")
        
        try:
            # Debug: Print available columns in original data
            print(f"🔍 Available columns in original data: {list(self.original_df.columns)}")
            
            # Use the original function
            self.xg_step_data = df_match_xg_step(self.original_df, meta=self.meta, per_bounds=self.per_bounds)
            
            if self.xg_step_data.empty:
                print("❌ No xG step data generated")
                return False
            
            # Debug: Print available columns in xG step data
            print(f"🔍 Available columns in xG step data: {list(self.xg_step_data.columns)}")
            
            # Count different types
            types = self.xg_step_data['type'].value_counts()
            print(f"✅ Generated xG step data:")
            for t, count in types.items():
                print(f"   {t}: {count} points")
                
            return True
            
        except Exception as e:
            print(f"❌ Error generating xG step data: {e}")
            import traceback
            print(f"🔍 Full traceback:")
            traceback.print_exc()
            return False
    
    def create_frontend_output(self):
        """Create simplified output for frontend team (matching original format exactly)"""
        print("📊 Creating frontend-friendly output...")
        
        if self.xg_step_data is None or self.xg_step_data.empty:
            print("❌ No xG step data available")
            return None
            
        # Create simplified structure (matching original xG step format exactly)
        output = {
            "match_id": str(self.match_id),
            "chart_info": {
                "x_axis": "Time (0-100%)",
                "y_axis": "Cumulative xG (0-100%)",
                "max_xg": float(self.meta['D'].iloc[0]) if not self.meta.empty else 0.0
            },
            "data": {
                "steps": [],
                "goals": [],
                "ticks": [],
                "y_ticks": []
            }
        }
        
        # Process each data point by type (using available columns from df_match_xg_step)
        for _, row in self.xg_step_data.iterrows():
            data_point = {
                "type": str(row["type"]),
                "x": float(row["x"]) if pd.notna(row["x"]) else None,
                "y": float(row["y"]) if pd.notna(row["y"]) else None,
                "team_id": int(row["team_id"]) if pd.notna(row["team_id"]) else None,
                "home": str(row["home"]) if pd.notna(row["home"]) else None,
                "label": str(row["label"]) if pd.notna(row["label"]) else None,
                "line_color": str(row["line_color"]) if pd.notna(row["line_color"]) else None,
                "point_color": str(row["point_color"]) if pd.notna(row["point_color"]) else None,
                "player_name": str(row["player_name"]) if pd.notna(row["player_name"]) else None,
                "minute_label": str(row["minute_label"]) if pd.notna(row["minute_label"]) else None
            }
            
            # Add to appropriate category
            if row["type"] == "shot":
                output["data"]["steps"].append(data_point)
            elif row["type"] == "goal":
                output["data"]["goals"].append(data_point)
            elif row["type"] == "tick":
                output["data"]["ticks"].append(data_point)
            elif row["type"] == "ytick":
                output["data"]["y_ticks"].append(data_point)
        
        print(f"✅ Created frontend data:")
        print(f"   Steps: {len(output['data']['steps'])}")
        print(f"   Goals: {len(output['data']['goals'])}")
        print(f"   Ticks: {len(output['data']['ticks'])}")
        print(f"   Y-ticks: {len(output['data']['y_ticks'])}")
        
        return output
    
    def export_to_json(self, output_data, filename=None):
        """Export data to JSON file"""
        if filename is None:
            filename = f"xg_step_data_{self.match_id}.json"
            
        filepath = Path(__file__).parent / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error exporting JSON: {e}")
            return None
    
    def print_summary(self, output_data):
        """Print summary of generated data"""
        print("\n" + "="*60)
        print("📊 xG STEP DATA SUMMARY")
        print("="*60)
        
        if not output_data or not output_data.get("data"):
            print("❌ No data to summarize")
            return
            
        data = output_data["data"]
        steps = data["steps"]
        goals = data["goals"]
        
        print(f"🎯 Match ID: {output_data['match_id']}")
        print(f"📈 Step Points: {len(steps)}")
        print(f"⚽ Goals: {len(goals)}")
        print(f"⏰ Time Ticks: {len(data['ticks'])}")
        print(f"📏 Y-axis Ticks: {len(data['y_ticks'])}")
        
        # Team breakdown
        if steps:
            teams = {}
            for step in steps:
                if step["team_id"] is not None:
                    team_id = step["team_id"]
                    if team_id not in teams:
                        teams[team_id] = {"home": step["home"], "steps": 0}
                    teams[team_id]["steps"] += 1
            
            print(f"\n👥 TEAM BREAKDOWN:")
            for team_id, stats in teams.items():
                print(f"   Team {team_id} ({stats['home']}): {stats['steps']} step points")
        
        # Goal summary
        if goals:
            print(f"\n⚽ GOAL SUMMARY:")
            for goal in goals:
                team_info = f"Team {goal['team_id']} ({goal['home']})"
                minute = goal.get('minute_label', 'Unknown')
                player = goal.get('player_name', 'Unknown')
                print(f"   {team_info}: {minute} - {player}")
        
        # xG summary
        max_xg = output_data["chart_info"]["max_xg"]
        print(f"\n📊 xG CHART INFO:")
        print(f"   Max xG scale: {max_xg}")
        print(f"   X-axis: {output_data['chart_info']['x_axis']}")
        print(f"   Y-axis: {output_data['chart_info']['y_axis']}")
    
    def show_xg_step_figure(self):
        """Display the xG step figure (uncomment to use)"""
        print("🎨 Displaying xG step figure...")
        
        if self.xg_step_data is None or self.xg_step_data.empty:
            print("❌ No xG step data to display")
            return
            
        try:
            # Create figure matching original
            fig, ax = plt.subplots(figsize=(10, 4), facecolor=futicolor.dark)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            # Y-axis guides behind (matching original)
            yticks = self.xg_step_data[self.xg_step_data['type'] == 'ytick']
            if not yticks.empty:
                for _, r in yticks.iterrows():
                    ax.axhline(r["y"], lw=0.6, color="#333333", zorder=0)
            
            # Time ticks behind (matching original)
            ticks = self.xg_step_data[self.xg_step_data['type'] == 'tick']
            if not ticks.empty:
                for _, r in ticks.iterrows():
                    ax.axvline(r["x"], lw=0.6, color="#555555", zorder=0)
                    if r["label"] == "period_end":
                        ax.text(r["x"], 98, "HT", ha="center", va="top", 
                               fontsize=8, color="#888888", zorder=4)
            
            # Step lines by team (matching original)
            steps = self.xg_step_data[self.xg_step_data['type'] == 'shot']
            if not steps.empty:
                for (_, tid), g in steps.groupby(["match_id", "team_id"], sort=False):
                    g = g.sort_values("x", kind="stable")
                    ax.plot(g["x"], g["y"], lw=2.0, color=g["line_color"].iloc[0],
                           drawstyle="steps-post", zorder=2)
            
            # Goal markers + labels (matching original)
            goals = self.xg_step_data[self.xg_step_data['type'] == 'goal']
            if not goals.empty:
                ax.scatter(goals["x"], goals["y"], s=100, c=goals["point_color"],
                          edgecolors=futicolor.dark, linewidths=1, zorder=3)
            
            # Add title
            ax.set_title(f"xG Step Chart - Match {self.match_id}", fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying figure: {e}")


if __name__ == "__main__":
    # Connect to Redis
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

    # Get all match event streams
    keys = r.keys("stream:match:*:events")
    match_ids = [key.split(":")[2] for key in keys]

    print(f"🔧 Found {len(match_ids)} matches in Redis")

    for match_id in match_ids:
        print("\n" + "="*60)
        print(f"📌 Generating xG step for match_id: {match_id}")
        print("="*60)

        generator = XGStepGeneratorRedis(match_id)

        if not generator.load_match_data():
            print(f"❌ Skipping match_id {match_id} (no data or no shot events)")
            continue

        if not generator.generate_xg_step_data():
            print(f"❌ Skipping match_id {match_id} (failed to generate xG step)")
            continue

        output_data = generator.create_frontend_output()
        if output_data is None:
            print(f"❌ Skipping match_id {match_id} (failed to create frontend output)")
            continue

        json_file = generator.export_to_json(output_data)
        if json_file is None:
            print(f"❌ Skipping match_id {match_id} (failed to export JSON)")
            continue

        generator.print_summary(output_data)
        
        # Uncomment the line below to show the xG step figure
        # generator.show_xg_step_figure()
        
        print(f"✅ xG step generation complete for match_id {match_id}")