#!/usr/bin/env python3
"""
Territory Generator - Redis Stream Version
==========================================

This script generates territory plot data for matches using Redis streams as data source,
following the same pattern as momentum_generator.py and other Redis generators.
It exports frontend-friendly JSON for easy consumption by the frontend team.

Usage:
    python territory_generator_redis.py

Output:
    - JSON files with territory data for each match
    - Optional figure display (uncomment show_territory_figure())
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import redis

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from futiplot.match_card.match_card_charts import df_match_territory
    from futiplot.soccer.pitch import plot_pitch
    from futiplot.utils import futicolor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class TerritoryGeneratorRedis:
    """Generate territory plot data for a specific match_id using Redis streams"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.territory_data = None
        self.original_df = None
        
        # Zone definitions (EXACTLY as in df_match_territory)
        self.length_zones = np.array([0.0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0], dtype=float)
        self.width_zones = np.array([0.0, 13.84, 24.84, 43.16, 54.16, 68.0], dtype=float)
        
        # Grid dimensions
        self.nx, self.ny = len(self.length_zones) - 1, len(self.width_zones) - 1
        
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

            # Ensure required columns exist for territory
            required_cols = ['type_name', 'x_start', 'y_start', 'team_id', 'phase_team']
            missing_cols = [col for col in required_cols if col not in df_game.columns]
            
            if missing_cols:
                print(f"⚠️ Missing required columns for territory: {missing_cols}")
                # Try to create phase_team from team_id if missing
                if 'phase_team' in missing_cols and 'team_id' in df_game.columns:
                    df_game['phase_team'] = df_game['team_id']
                    missing_cols.remove('phase_team')
                    print(f"✅ Created phase_team from team_id")
                
                if missing_cols:
                    print(f"❌ Still missing required columns: {missing_cols}")
                    return False

            # Filter possession events for territory calculation
            possession_actions = {"pass", "dribble", "carry", "reception"}
            
            # Lowercase type_name for consistent filtering
            df_game['type_name'] = df_game['type_name'].astype(str).str.lower()
            possession_events = df_game[df_game["type_name"].isin(possession_actions)]

            print(f"🏟️ Found {len(possession_events)} possession events for territory (before deduplication)")

            # Deduplicate possession events to avoid artificial territory dominance
            if not possession_events.empty:
                events_before = len(possession_events)
                
                # Try to identify if we have a unique event ID column
                id_cols = ['action_id', 'event_id', 'original_event_id']
                unique_id_col = None
                for col in id_cols:
                    if col in possession_events.columns:
                        unique_id_col = col
                        break
                
                if unique_id_col:
                    # If we have unique IDs, use those for deduplication (safest)
                    possession_events = possession_events.drop_duplicates(subset=[unique_id_col], keep='first')
                    print(f"🔧 Deduplicated by {unique_id_col}: {events_before} → {len(possession_events)} (removed {events_before - len(possession_events)} exact duplicates)")
                else:
                    # Fallback: Only remove EXACT duplicates (no rounding)
                    exact_dedup_cols = ['team_id', 'time_seconds', 'x_start', 'y_start', 'type_name', 'player_id', 'phase_team']
                    available_exact_cols = [col for col in exact_dedup_cols if col in possession_events.columns]
                    
                    if available_exact_cols:
                        possession_events = possession_events.drop_duplicates(subset=available_exact_cols, keep='first')
                        print(f"🔧 Deduplicated by exact match: {events_before} → {len(possession_events)} (removed {events_before - len(possession_events)} exact duplicates)")
                    else:
                        print(f"⚠️ No suitable columns for deduplication, keeping all {events_before} events")
                
                # Update the main dataframe with deduplicated events
                non_possession_events = df_game[~df_game["type_name"].isin(possession_actions)]
                df_game = pd.concat([non_possession_events, possession_events], ignore_index=True).sort_values('time_seconds', na_position='last')

            print(f"🏟️ Final possession events: {len(possession_events)}")

            if possession_events.empty:
                print("⚠️ No possession events found in this match")
                return False

            self.original_df = df_game

            # Convert numeric columns needed for territory generation
            numeric_cols = [
                'x_start', 'y_start', 'team_id', 'phase_team', 'period_id', 
                'time_seconds', 'player_id', 'game_id', 'match_id', 'phase_id',
                'possession_id', 'sequence_id'
            ]
            for col in numeric_cols:
                if col in df_game.columns:
                    df_game[col] = pd.to_numeric(df_game[col], errors='coerce')
            
            # Ensure string columns are properly typed
            string_cols = ['type_name', 'home', 'result_name']
            for col in string_cols:
                if col in df_game.columns:
                    df_game[col] = df_game[col].astype(str)
            
            # Debug: Check data types after conversion
            print(f"🔍 Data types after conversion:")
            for col in ['x_start', 'y_start', 'team_id', 'phase_team', 'home']:
                if col in df_game.columns:
                    print(f"   {col}: {df_game[col].dtype} (sample: {df_game[col].iloc[0] if len(df_game) > 0 else 'N/A'})")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def generate_territory_data(self):
        """Generate territory data using the original function"""
        print("🏟️ Generating territory data...")
        
        try:
            # Debug: Print available columns in original data
            print(f"🔍 Available columns in original data: {list(self.original_df.columns)}")
            
            # Additional data cleaning before calling df_match_territory
            df_clean = self.original_df.copy()
            
            # Create a minimal dataframe with only the columns needed for territory calculation
            required_cols = ['x_start', 'y_start', 'team_id', 'phase_team', 'home', 'type_name', 'match_id']
            
            # Ensure all required columns exist
            for col in required_cols:
                if col not in df_clean.columns:
                    if col == 'match_id':
                        df_clean['match_id'] = self.match_id
                    elif col == 'phase_team' and 'team_id' in df_clean.columns:
                        df_clean['phase_team'] = df_clean['team_id']
                    else:
                        print(f"❌ Missing required column: {col}")
                        return False
            
            # Keep only required columns to avoid any data type issues with other columns
            df_clean = df_clean[required_cols].copy()
            
            # Clean numeric columns
            numeric_cols = ['x_start', 'y_start', 'team_id', 'phase_team', 'match_id']
            for col in numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Clean string columns
            string_cols = ['home', 'type_name']
            for col in string_cols:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
            
            # Remove rows with invalid data
            df_clean = df_clean.dropna(subset=['x_start', 'y_start', 'team_id', 'phase_team'])
            
            # Ensure coordinates are within valid pitch bounds
            df_clean = df_clean[
                (df_clean['x_start'] >= 0) & (df_clean['x_start'] <= 105) &
                (df_clean['y_start'] >= 0) & (df_clean['y_start'] <= 68)
            ]
            
            # Ensure team_id and phase_team are clean integers
            for col in ['team_id', 'phase_team', 'match_id']:
                df_clean[col] = df_clean[col].astype('int64')
            
            # Ensure home column only contains 'home' or 'away'
            df_clean = df_clean[df_clean['home'].isin(['home', 'away'])]
            
            # Add any missing columns that df_match_territory might expect
            if 'game_id' not in df_clean.columns:
                df_clean['game_id'] = df_clean['match_id']
            
            # Debug: Check final data types and sample values
            print(f"🔍 Final data check before df_match_territory:")
            for col in ['x_start', 'y_start', 'team_id', 'phase_team', 'home', 'type_name']:
                if col in df_clean.columns:
                    dtype = df_clean[col].dtype
                    sample = df_clean[col].iloc[0] if len(df_clean) > 0 else 'N/A'
                    print(f"   {col}: {dtype} (sample: {sample})")
            
            print(f"🔍 Clean dataframe shape: {df_clean.shape}")
            
            # Use a custom wrapper to handle the string concatenation issue
            self.territory_data = self._safe_df_match_territory(df_clean)
            
            if self.territory_data.empty:
                print("❌ No territory data generated")
                return False
            
            # Debug: Print available columns in territory data
            print(f"🔍 Available columns in territory data: {list(self.territory_data.columns)}")
            
            print(f"✅ Generated territory data: {len(self.territory_data)} zones")
            return True
            
        except Exception as e:
            print(f"❌ Error generating territory data: {e}")
            import traceback
            print(f"🔍 Full traceback:")
            traceback.print_exc()
            return False
    
    def _safe_df_match_territory(self, df):
        """
        Safe wrapper around df_match_territory that handles the string concatenation issue
        """
        try:
            # Try the original function first
            return df_match_territory(df)
        except Exception as e:
            if "ufunc 'add' did not contain a loop" in str(e):
                print("🔧 Handling string concatenation issue with custom implementation...")
                return self._custom_territory_calculation(df)
            else:
                raise e
    
    def _custom_territory_calculation(self, df):
        """
        Custom territory calculation that avoids the NumPy string concatenation issue
        """
        # Filter to possession events only
        possession_actions = {"pass", "dribble", "carry", "reception"}
        df_poss = df[df['type_name'].str.lower().isin(possession_actions)].copy()
        
        if df_poss.empty:
            return pd.DataFrame()
        
        # Create zone bins
        x_bins = pd.cut(df_poss['x_start'], bins=self.length_zones, include_lowest=True, labels=False)
        y_bins = pd.cut(df_poss['y_start'], bins=self.width_zones, include_lowest=True, labels=False)
        
        # Group by zone and team
        df_poss['x_bin'] = x_bins
        df_poss['y_bin'] = y_bins
        df_poss = df_poss.dropna(subset=['x_bin', 'y_bin'])
        
        # Count possessions per zone per team
        zone_counts = df_poss.groupby(['x_bin', 'y_bin', 'phase_team']).size().reset_index(name='count')
        
        # Calculate total possessions per zone
        zone_totals = zone_counts.groupby(['x_bin', 'y_bin'])['count'].sum().reset_index(name='total_count')
        
        # Merge to get percentages
        zone_counts = zone_counts.merge(zone_totals, on=['x_bin', 'y_bin'])
        zone_counts['possession_pct'] = zone_counts['count'] / zone_counts['total_count']
        
        # Find dominant team per zone
        dominant_zones = zone_counts.loc[zone_counts.groupby(['x_bin', 'y_bin'])['possession_pct'].idxmax()]
        
        # Create output dataframe
        result_zones = []
        
        for _, zone in dominant_zones.iterrows():
            x_bin = int(zone['x_bin'])
            y_bin = int(zone['y_bin'])
            
            # Get zone boundaries
            x_min = self.length_zones[x_bin]
            x_max = self.length_zones[x_bin + 1]
            y_min = self.width_zones[y_bin]
            y_max = self.width_zones[y_bin + 1]
            
            # Determine team info
            team_id = zone['phase_team']
            possession_pct = zone['possession_pct']
            
            # Get team home/away status
            team_info = df[df['team_id'] == team_id]['home'].iloc[0] if len(df[df['team_id'] == team_id]) > 0 else 'home'
            
            # Only include zones with significant possession (>55%)
            if possession_pct > 0.55:
                # Create possession label safely
                pct_int = max(0, min(100, int(round(possession_pct * 100))))
                possession_label = f"{pct_int}%"
                
                # Assign colors based on team
                if team_info == 'home':
                    fill_color = futicolor.blue if possession_pct > 0.65 else futicolor.blue1
                    team_label = 'home'
                else:
                    fill_color = futicolor.pink if possession_pct > 0.65 else futicolor.pink1
                    team_label = 'away'
                
                result_zones.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'team_label': team_label,
                    'possession_label': possession_label,
                    'fill_color': fill_color
                })
        
        return pd.DataFrame(result_zones)
    
    def create_frontend_output(self):
        """Create simplified output for frontend team (matching original format exactly)"""
        print("📊 Creating frontend-friendly output...")
        
        if self.territory_data is None or self.territory_data.empty:
            print("❌ No territory data available")
            return None
            
        # Create simplified structure (matching original territory format exactly)
        output = {
            "match_id": str(self.match_id),
            "pitch_info": {
                "length_meters": 105.0,
                "width_meters": 68.0,
                "grid_size": f"{self.nx} x {self.ny}",
                "total_zones": self.nx * self.ny
            },
            "zones": []
        }
        
        # Create a complete grid (including zones with no data)
        for y_bin in range(self.ny):
            for x_bin in range(self.nx):
                zone_key = f"({x_bin},{y_bin})"
                
                # Get zone boundaries
                x_min = self.length_zones[x_bin]
                x_max = self.length_zones[x_bin + 1]
                y_min = self.width_zones[y_bin]
                y_max = self.width_zones[y_bin + 1]
                
                # Find matching territory data for this zone
                zone_data = None
                for _, row in self.territory_data.iterrows():
                    if (abs(row["x_min"] - x_min) < 0.1 and 
                        abs(row["x_max"] - x_max) < 0.1 and
                        abs(row["y_min"] - y_min) < 0.1 and
                        abs(row["y_max"] - y_max) < 0.1):
                        zone_data = row
                        break
                
                # Create zone entry with ALL data points from df_match_territory
                zone_entry = {
                    # Grid coordinates (for indexing)
                    "zone_id": zone_key,
                    "x": x_bin,
                    "y": y_bin,
                    
                    # Zone boundaries (from df_match_territory)
                    "x_min": float(x_min),
                    "x_max": float(x_max),
                    "y_min": float(y_min),
                    "y_max": float(y_max),
                    
                    # Default values (for zones with no data)
                    "team_label": "neutral",
                    "possession_label": "50%",
                    "home_value": 50,
                    "away_value": 50,
                    "fill_color": "#cccccc"
                }
                
                if zone_data is not None:
                    # Zone has possession data - include ALL original data points
                    possession_str = zone_data["possession_label"]
                    possession_pct = int(possession_str.replace('%', ''))
                    
                    # Update with complete data from df_match_territory
                    zone_entry.update({
                        # Original data points from df_match_territory
                        "team_label": str(zone_data["team_label"]),
                        "possession_label": str(zone_data["possession_label"]),
                        "fill_color": str(zone_data["fill_color"]),
                        
                        # Calculated values (for backward compatibility)
                        "home_value": possession_pct if zone_data["team_label"] == "home" else 100 - possession_pct,
                        "away_value": 100 - possession_pct if zone_data["team_label"] == "home" else possession_pct
                    })
                
                output["zones"].append(zone_entry)
        
        print(f"✅ Created frontend data with {len(output['zones'])} zones")
        return output
    
    def export_to_json(self, output_data, filename=None):
        """Export data to JSON file"""
        if filename is None:
            filename = f"territory_data_{self.match_id}.json"
            
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
        print("📊 TERRITORY DATA SUMMARY")
        print("="*60)
        
        if not output_data or not output_data.get("zones"):
            print("❌ No data to summarize")
            return
            
        zones = output_data["zones"]
        total_zones = len(zones)
        
        print(f"🎯 Match ID: {output_data['match_id']}")
        print(f"📏 Grid Size: {output_data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {total_zones}")
        
        # Team breakdown
        team_counts = {"home": 0, "away": 0, "neutral": 0}
        for zone in zones:
            team_label = zone.get("team_label", "neutral")
            if team_label in team_counts:
                team_counts[team_label] += 1
        
        print(f"\n👥 TEAM BREAKDOWN:")
        for team, count in team_counts.items():
            percentage = (count / total_zones) * 100 if total_zones > 0 else 0
            print(f"   {team.capitalize()}: {count} zones ({percentage:.1f}%)")
        
        # Sample zones with ALL data points
        print(f"\n🎯 SAMPLE ZONES:")
        sample_zones = zones[:3]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']}:")
            print(f"     Grid: x={zone['x']}, y={zone['y']}")
            print(f"     Boundaries: x_min={zone['x_min']:.1f}, x_max={zone['x_max']:.1f}, y_min={zone['y_min']:.1f}, y_max={zone['y_max']:.1f}")
            print(f"     Team: {zone['team_label']}")
            print(f"     Possession: {zone['possession_label']}")
            print(f"     Values: home={zone['home_value']}%, away={zone['away_value']}%")
            print(f"     Color: {zone['fill_color']}")
            print()
    
    def show_territory_figure(self):
        """Display the territory plot figure (uncomment to use)"""
        print("🎨 Displaying territory plot figure...")
        
        if self.territory_data is None or self.territory_data.empty:
            print("❌ No territory data to display")
            return
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            
            # Create pitch
            fig, ax, pitch = plot_pitch(orientation="wide", figsize=(12, 8), linewidth=2)
            
            # Plot each zone
            for _, zone in self.territory_data.iterrows():
                x_min, x_max = zone["x_min"], zone["x_max"]
                y_min, y_max = zone["y_min"], zone["y_max"]
                color = zone["fill_color"]
                label = zone["possession_label"]
                
                # Create rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                               facecolor=color, alpha=0.7, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
                
                # Add text label
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                ax.text(center_x, center_y, label, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
            
            # Set title and labels
            ax.set_title(f"Territory Plot - Match {self.match_id}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Length (meters)", fontsize=12)
            ax.set_ylabel("Width (meters)", fontsize=12)
            
            # Set axis limits
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            
            # Invert y-axis for proper football view
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError as e:
            print(f"❌ Could not import plotting libraries: {e}")
            print("💡 To enable plotting, install: matplotlib")
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
        print(f"📌 Generating territory for match_id: {match_id}")
        print("="*60)

        generator = TerritoryGeneratorRedis(match_id)

        if not generator.load_match_data():
            print(f"❌ Skipping match_id {match_id} (no data or no possession events)")
            continue

        if not generator.generate_territory_data():
            print(f"❌ Skipping match_id {match_id} (failed to generate territory)")
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
        
        # Uncomment the line below to show the territory figure
        # generator.show_territory_figure()
        
        print(f"✅ Territory generation complete for match_id {match_id}")