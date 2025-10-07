#!/usr/bin/env python3
"""
Shotmap Generator - Redis Stream Version
========================================

This script generates shotmap data for matches using Redis streams as data source,
following the same pattern as momentum_generator.py. It exports frontend-friendly JSON
for easy consumption by the frontend team.

Usage:
    python shotmap_generator.py

Output:
    - JSON files with shotmap data for each match
    - Optional figure display (uncomment show_shotmap_figure())
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
    from futiplot.match_card.match_card_charts import df_match_shotmap
    from futiplot.soccer.pitch import plot_pitch
    from futiplot.utils import futicolor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class ShotmapGenerator:
    """Generate shotmap data for a specific match_id using Redis streams"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.shotmap_data = None
        self.original_df = None
        
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

            # Ensure required columns exist for shotmap
            required_cols = ['type_name', 'x_start', 'y_start', 'team_id']
            missing_cols = [col for col in required_cols if col not in df_game.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns for shotmap: {missing_cols}")
                return False

            # Filter shot events (events with xG > 0 or type_name = 'shot')
            shot_events = df_game[
                (df_game.get('xg', 0) > 0) | 
                (df_game['type_name'].astype(str).str.lower() == 'shot')
            ]

            print(f"🏹 Found {len(shot_events)} shot events")

            if shot_events.empty:
                print("⚠️ No shot events found in this match")
                return False

            self.original_df = df_game

            # Convert numeric columns needed for shotmap generation
            numeric_cols = ['x_start', 'y_start', 'team_id', 'player_id', 'xg']
            for col in numeric_cols:
                if col in df_game.columns:
                    df_game[col] = pd.to_numeric(df_game[col], errors='coerce')

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_shotmap_data(self):
        """Generate shotmap data using the original function"""
        print("🎯 Generating shotmap data...")
        
        try:
            # Use the original function
            self.shotmap_data = df_match_shotmap(self.original_df)
            
            if self.shotmap_data.empty:
                print("❌ No shotmap data generated")
                return False
                
            print(f"✅ Generated {len(self.shotmap_data)} shot data points")
            return True
            
        except Exception as e:
            print(f"❌ Error generating shotmap data: {e}")
            return False
    
    def create_frontend_output(self):
        """Create simplified output for frontend team"""
        print("📊 Creating frontend-friendly output...")
        
        if self.shotmap_data is None or self.shotmap_data.empty:
            print("❌ No shotmap data available")
            return None
            
        # Create simplified structure
        output = {
            "match_id": str(self.match_id),
            "pitch_info": {
                "length": 105.0,
                "width": 68.0,
                "orientation": "wide"
            },
            "shots": []
        }
        
        # Process each shot
        for _, shot in self.shotmap_data.iterrows():
            shot_data = {
                "team_id": int(shot["team_id"]) if pd.notna(shot["team_id"]) else None,
                "home": str(shot["home"]) if pd.notna(shot["home"]) else None,
                "x": float(shot["x_both"]) if pd.notna(shot["x_both"]) else None,
                "y": float(shot["y_both"]) if pd.notna(shot["y_both"]) else None,
                "xg": float(shot["xg"]) if pd.notna(shot["xg"]) else 0.0,
                "goal": int(shot["goal"]) if pd.notna(shot["goal"]) else 0,
                "size": float(shot["shot_size"]) if pd.notna(shot["shot_size"]) else 1.0,
                "point_color": str(shot["point_color"]) if pd.notna(shot["point_color"]) else "#000000",
                "edge_color": str(shot["edge_color"]) if pd.notna(shot["edge_color"]) else "#000000"
            }
            output["shots"].append(shot_data)
        
        print(f"✅ Created {len(output['shots'])} shot data points")
        return output
    
    def export_to_json(self, output_data, filename=None):
        """Export data to JSON file"""
        if filename is None:
            filename = f"shotmap_data_{self.match_id}.json"
            
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
        print("📊 SHOTMAP DATA SUMMARY")
        print("="*60)
        
        if not output_data or not output_data.get("shots"):
            print("❌ No data to summarize")
            return
            
        shots = output_data["shots"]
        total_shots = len(shots)
        goals = sum(1 for shot in shots if shot["goal"] == 1)
        
        print(f"🎯 Match ID: {output_data['match_id']}")
        print(f"🏹 Total Shots: {total_shots}")
        print(f"⚽ Goals: {goals}")
        print(f"📈 Goal Rate: {goals/total_shots*100:.1f}%" if total_shots > 0 else "N/A")
        
        # Team breakdown
        teams = {}
        for shot in shots:
            team_id = shot["team_id"]
            if team_id is not None and team_id not in teams:
                teams[team_id] = {"shots": 0, "goals": 0, "home": shot["home"]}
            if team_id is not None:
                teams[team_id]["shots"] += 1
                if shot["goal"] == 1:
                    teams[team_id]["goals"] += 1
        
        print(f"\n👥 TEAM BREAKDOWN:")
        for team_id, stats in teams.items():
            print(f"   Team {team_id} ({stats['home']}): {stats['shots']} shots, {stats['goals']} goals")
        
        # xG summary
        total_xg = sum(shot["xg"] for shot in shots)
        print(f"\n📊 xG SUMMARY:")
        print(f"   Total xG: {total_xg:.2f}")
        print(f"   Average xG per shot: {total_xg/total_shots:.3f}" if total_shots > 0 else "N/A")
        
        # Size distribution
        sizes = [shot["size"] for shot in shots]
        print(f"\n📏 SIZE DISTRIBUTION:")
        print(f"   Min size: {min(sizes):.3f}")
        print(f"   Max size: {max(sizes):.3f}")
        print(f"   Average size: {np.mean(sizes):.3f}")
    
    def show_shotmap_figure(self):
        """Display the shotmap figure (uncomment to use)"""
        print("🎨 Displaying shotmap figure...")
        
        if self.shotmap_data is None or self.shotmap_data.empty:
            print("❌ No shotmap data to display")
            return
            
        try:
            # Create figure with pitch
            fig, ax, pitch = plot_pitch(orientation="wide", figsize=(12, 8), linewidth=2)
            
            # Plot shots
            for _, shot in self.shotmap_data.iterrows():
                x, y = shot["x_both"], shot["y_both"]
                size = shot["shot_size"] * 200  # Scale size
                color = shot["point_color"]
                edge_color = shot["edge_color"]
                goal = shot["goal"]
                
                # Different marker for goals
                marker = "o"  # Use circles for all shots
                size = size * 1.5 if goal == 1 else size  # Larger for goals
                
                ax.scatter(x, y, s=size, c=color, edgecolors=edge_color, 
                          linewidth=1, marker=marker, alpha=0.8)
            
            # Add title
            ax.set_title(f"Shotmap - Match {self.match_id}", fontsize=16, fontweight='bold')
            
            # Add legend
            home_shots = self.shotmap_data[self.shotmap_data["home"] == "home"]
            away_shots = self.shotmap_data[self.shotmap_data["home"] == "away"]
            
            if not home_shots.empty:
                ax.scatter([], [], c=futicolor.blue, s=100, label="Home Shots", alpha=0.8)
            if not away_shots.empty:
                ax.scatter([], [], c=futicolor.pink, s=100, label="Away Shots", alpha=0.8)
            
            ax.legend(loc='upper right')
            
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
        print(f"📌 Generating shotmap for match_id: {match_id}")
        print("="*60)

        generator = ShotmapGenerator(match_id)

        if not generator.load_match_data():
            print(f"❌ Skipping match_id {match_id} (no data or no shot events)")
            continue

        if not generator.generate_shotmap_data():
            print(f"❌ Skipping match_id {match_id} (failed to generate shotmap)")
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
        
        # Uncomment the line below to show the shotmap figure
        # generator.show_shotmap_figure()
        
        print(f"✅ Shotmap generation complete for match_id {match_id}")
