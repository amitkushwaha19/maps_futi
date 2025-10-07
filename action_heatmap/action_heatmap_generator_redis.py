#!/usr/bin/env python3
"""
Action Heatmap Generator - Redis Stream Version
===============================================

This script generates action heatmap data for matches using Redis streams as data source,
following the same pattern as momentum_generator.py. It exports frontend-friendly JSON
for easy consumption by the frontend team.

Usage:
    python action_heatmap_generator_redis.py

Output:
    - JSON files with action heatmap data for each match
    - Optional figure display (uncomment show_heatmap_figure())
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import redis

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from futiplot.match_card.match_card_charts import df_match_action_heatmap
    from futiplot.utils import futicolor
    from futiplot.soccer.pitch import plot_pitch
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class ActionHeatmapGenerator:
    """Generate action heatmap data for a specific match_id using Redis streams"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.heatmap_data = None
        self.original_df = None
        
        # Grid dimensions (same as in df_match_action_heatmap)
        self.x_zones = 21  # 20 cells along length
        self.y_zones = 14  # 13 cells along width
        self.total_zones = 20 * 13  # 260 zones
        
        # Pitch dimensions
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
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

            # Ensure required columns exist for heatmap
            required_cols = ['type_name', 'x_start', 'y_start', 'team_id']
            missing_cols = [col for col in required_cols if col not in df_game.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns for heatmap: {missing_cols}")
                return False

            # Filter action events (broader than possession events)
            action_types = {
                "pass", "dribble", "carry", "reception", "shot", "tackle", 
                "interception", "clearance", "cross", "corner", "throw_in"
            }
            
            # Lowercase type_name for consistent filtering
            df_game['type_name'] = df_game['type_name'].astype(str).str.lower()
            action_events = df_game[df_game["type_name"].isin(action_types)]

            print(f"🎯 Found {len(action_events)} action events for heatmap (before deduplication)")

            # Deduplicate action events to avoid artificial heatmap intensity
            if not action_events.empty:
                events_before = len(action_events)
                
                # Try to identify if we have a unique event ID column
                id_cols = ['action_id', 'event_id', 'original_event_id']
                unique_id_col = None
                for col in id_cols:
                    if col in action_events.columns:
                        unique_id_col = col
                        break
                
                if unique_id_col:
                    # If we have unique IDs, use those for deduplication (safest)
                    action_events = action_events.drop_duplicates(subset=[unique_id_col], keep='first')
                    print(f"🔧 Deduplicated by {unique_id_col}: {events_before} → {len(action_events)} (removed {events_before - len(action_events)} exact duplicates)")
                else:
                    # Fallback: Only remove EXACT duplicates (no rounding)
                    exact_dedup_cols = ['team_id', 'time_seconds', 'x_start', 'y_start', 'type_name', 'player_id']
                    available_exact_cols = [col for col in exact_dedup_cols if col in action_events.columns]
                    
                    if available_exact_cols:
                        action_events = action_events.drop_duplicates(subset=available_exact_cols, keep='first')
                        print(f"🔧 Deduplicated by exact match: {events_before} → {len(action_events)} (removed {events_before - len(action_events)} exact duplicates)")
                    else:
                        print(f"⚠️ No suitable columns for deduplication, keeping all {events_before} events")
                
                # Update the main dataframe with deduplicated events
                non_action_events = df_game[~df_game["type_name"].isin(action_types)]
                df_game = pd.concat([non_action_events, action_events], ignore_index=True).sort_values('time_seconds', na_position='last')

            print(f"🎯 Final action events: {len(action_events)}")

            if action_events.empty:
                print("⚠️ No action events found in this match")
                return False

            self.original_df = action_events

            # Convert numeric columns needed for heatmap generation
            numeric_cols = ['x_start', 'y_start', 'team_id', 'player_id']
            for col in numeric_cols:
                if col in df_game.columns:
                    df_game[col] = pd.to_numeric(df_game[col], errors='coerce')

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def generate_heatmap_data(self):
        """Generate heatmap data using the original function"""
        print("📊 Generating action heatmap data...")
        
        try:
            # Use the original function with same parameters
            self.heatmap_data = df_match_action_heatmap(
                self.original_df,
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                length_edges=self.x_zones,
                width_edges=self.y_zones
            )
            
            if self.heatmap_data.empty:
                print("❌ No heatmap data generated")
                return False
                
            # Debug: Print available columns
            print(f"🔍 Available columns in heatmap data: {list(self.heatmap_data.columns)}")
            
            # Count zones and teams
            total_zones = len(self.heatmap_data)
            teams = self.heatmap_data['team_id'].nunique() if 'team_id' in self.heatmap_data.columns else 0
            
            print(f"✅ Generated heatmap data:")
            print(f"   Total zones: {total_zones}")
            print(f"   Teams: {teams}")
                
            return True
            
        except Exception as e:
            print(f"❌ Error generating heatmap data: {e}")
            return False
    
    def create_frontend_output(self):
        """Create simplified output for frontend team"""
        print("📊 Creating frontend-friendly output...")
        
        if self.heatmap_data is None or self.heatmap_data.empty:
            print("❌ No heatmap data available")
            return None
            
        # Create simplified structure (matching original format exactly)
        output = {
            "match_id": str(self.match_id),
            "pitch_info": {
                "length_meters": self.pitch_length,
                "width_meters": self.pitch_width,
                "grid_size": f"{self.x_zones-1} x {self.y_zones-1}",
                "total_zones": self.total_zones
            },
            "zones": []
        }
        
        # Process each zone and create separate entries for home and away teams
        # Since the function returns both_count, home_count, away_count, we need to create separate zones
        
        # First, collect all unique team IDs from the original data to assign proper team_ids
        team_ids = self.original_df['team_id'].unique()
        home_team_id = None
        away_team_id = None
        
        # Determine which team is home/away
        for team_id in team_ids:
            team_data = self.original_df[self.original_df['team_id'] == team_id]
            if team_data['home'].iloc[0] == 'home':
                home_team_id = str(int(team_id))
            else:
                away_team_id = str(int(team_id))
        
        # Process each zone and create entries for both teams
        for _, zone in self.heatmap_data.iterrows():
            zone_id = f"({zone['x_min']:.1f},{zone['y_min']:.1f})"
            center_x = float((zone["x_min"] + zone["x_max"]) / 2)
            center_y = float((zone["y_min"] + zone["y_max"]) / 2)
            
            # Create entry for home team if it has activity in this zone
            if zone["home_count"] > 0:
                home_zone_entry = {
                    "zone_id": zone_id,
                    "team_id": home_team_id,
                    "x_min": float(zone["x_min"]),
                    "x_max": float(zone["x_max"]),
                    "y_min": float(zone["y_min"]),
                    "y_max": float(zone["y_max"]),
                    "count": int(zone["home_count"]),
                    "intensity": float(zone["home_count"]) * 0.5,  # Transform count to intensity
                    "alpha": float(zone["home_fill"]),
                    "center_x": center_x,
                    "center_y": center_y
                }
                output["zones"].append(home_zone_entry)
            
            # Create entry for away team if it has activity in this zone
            if zone["away_count"] > 0:
                away_zone_entry = {
                    "zone_id": zone_id,
                    "team_id": away_team_id,
                    "x_min": float(zone["x_min"]),
                    "x_max": float(zone["x_max"]),
                    "y_min": float(zone["y_min"]),
                    "y_max": float(zone["y_max"]),
                    "count": int(zone["away_count"]),
                    "intensity": float(zone["away_count"]) * 0.5,  # Transform count to intensity
                    "alpha": float(zone["away_fill"]),
                    "center_x": center_x,
                    "center_y": center_y
                }
                output["zones"].append(away_zone_entry)
        
        # Add teams list to pitch_info (matching original format)
        team_list = []
        for zone in output["zones"]:
            team_id = zone["team_id"]
            if team_id is not None and team_id not in team_list:
                team_list.append(team_id)
        
        output["pitch_info"]["teams"] = team_list
        
        print(f"✅ Created frontend data:")
        print(f"   Total zones: {len(output['zones'])}")
        print(f"   Teams: {len(team_list)}")
        
        return output
    
    def export_to_json(self, output_data, filename=None):
        """Export data to JSON file"""
        if filename is None:
            filename = f"action_heatmap_data_{self.match_id}.json"
            
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
        print("📊 ACTION HEATMAP DATA SUMMARY")
        print("="*60)
        
        if not output_data or not output_data.get("zones"):
            print("❌ No data to summarize")
            return
            
        zones = output_data["zones"]
        teams = output_data["pitch_info"]["teams"]
        
        print(f"🎯 Match ID: {output_data['match_id']}")
        print(f"📏 Grid Size: {output_data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {len(zones)}")
        print(f"👥 Teams: {', '.join(teams)}")
        
        # Team breakdown (using correct column names that match original output)
        for team_id in teams:
            team_zones = [zone for zone in zones if zone["team_id"] == team_id]
            total_events = sum(zone["count"] for zone in team_zones)
            avg_intensity = np.mean([zone["intensity"] for zone in team_zones]) if team_zones else 0
            avg_alpha = np.mean([zone["alpha"] for zone in team_zones]) if team_zones else 0
            active_zones = len(team_zones)
            
            print(f"\n🏃 TEAM {team_id}:")
            print(f"   Active zones: {active_zones}")
            print(f"   Total events: {total_events}")
            print(f"   Average intensity: {avg_intensity:.3f}")
            print(f"   Average alpha: {avg_alpha:.3f}")
        
        # Sample zones (using correct column names that match original output)
        print(f"\n🎯 SAMPLE ZONES:")
        sample_zones = zones[:3]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']}: Team={zone['team_id']}, Count={zone['count']}, Intensity={zone['intensity']:.3f}, Alpha={zone['alpha']:.3f}")
    
    def show_heatmap_figure(self):
        """Display the action heatmap figure"""
        print("🎨 Displaying action heatmap figure...")
        
        if self.heatmap_data is None or self.heatmap_data.empty:
            print("❌ No heatmap data to display")
            return
            
        try:
            # Create pitch
            fig, ax, pitch = plot_pitch(orientation="wide", figsize=(12, 8), linewidth=2)
            
            # Plot each zone (using available columns from heatmap data)
            for _, zone in self.heatmap_data.iterrows():
                x_min, x_max = zone["x_min"], zone["x_max"]
                y_min, y_max = zone["y_min"], zone["y_max"]
                alpha = zone["both_fill"]  # Use both_fill as alpha
                count = zone["both_count"]  # Use both_count for labels
                
                # Create rounded rectangle with gap
                gap = 0.4
                w = x_max - x_min
                h = y_max - y_min
                w2 = max(0.0, w - gap)
                h2 = max(0.0, h - gap)
                
                if w2 > 0 and h2 > 0:
                    x2 = x_min + 0.5 * (w - w2)
                    y2 = y_min + 0.5 * (h - h2)
                    
                    r = min(w2, h2) * 0.18
                    r = max(0.0, min(r, 0.9))
                    
                    rect = FancyBboxPatch(
                        (x2, y2), w2, h2,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        facecolor=futicolor.purple,
                        edgecolor="none",
                        linewidth=0,
                        alpha=alpha,
                        zorder=3,
                        clip_on=True
                    )
                    ax.add_patch(rect)
                    
                    # Add count label for zones with significant activity
                    if count > 0:
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        ax.text(center_x, center_y, str(count), ha='center', va='center', 
                               fontsize=6, fontweight='bold', color='white', alpha=0.8)
            
            # Set title and labels
            ax.set_title(f"Action Heatmap - Match: {self.match_id}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Length (meters)", fontsize=12)
            ax.set_ylabel("Width (meters)", fontsize=12)
            
            # Set axis limits
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            
            # Invert y-axis for proper football view
            ax.invert_yaxis()
            
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
        print(f"📌 Generating action heatmap for match_id: {match_id}")
        print("="*60)

        generator = ActionHeatmapGenerator(match_id)

        if not generator.load_match_data():
            print(f"❌ Skipping match_id {match_id} (no data or no action events)")
            continue

        if not generator.generate_heatmap_data():
            print(f"❌ Skipping match_id {match_id} (failed to generate heatmap)")
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
        
        # Uncomment the line below to show the heatmap figure
        # generator.show_heatmap_figure()
        
        print(f"✅ Action heatmap generation complete for match_id {match_id}")