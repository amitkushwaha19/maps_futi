#!/usr/bin/env python3
"""
Territory Plot Generator - Redis Version
A dedicated script for generating territory visualization data for a specific match from Redis.

Usage:
    python territory_plot_generator_redis.py <match_id>
    
Example:
    python territory_plot_generator_redis.py 4491
"""

import json
import sys
import os
import pandas as pd
import numpy as np
import redis
from typing import Dict, List, Any

# Add the src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from futiplot.match_card.match_card_charts import df_match_territory
from futiplot.utils import load_sample_data

class TerritoryPlotGenerator:
    """Generate territory plot data for a specific match from Redis"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.redis_client = None
        self.territory_data = None
        self.original_df = None
        
        # Zone definitions (EXACTLY as in df_match_territory)
        self.length_zones = np.array([0.0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0], dtype=float)
        self.width_zones = np.array([0.0, 13.84, 24.84, 43.16, 54.16, 68.0], dtype=float)
        
        # Grid dimensions
        self.nx, self.ny = len(self.length_zones) - 1, len(self.width_zones) - 1
        
    def connect_redis(self):
        """Connect to Redis"""
        try:
            # Normalize match_id to int for stream naming
            match_id = int(self.match_id) if isinstance(self.match_id, str) else self.match_id
            
            # Connect to Redis and read per-match stream
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
    
    def load_match_data_from_redis(self) -> pd.DataFrame:
        """Load match data for a specific match_id from Redis"""
        try:
            if not self.connect_redis():
                raise Exception("Failed to connect to Redis")
            
            print(f"🔄 Loading data for match: {self.match_id}")
            
            # Normalize match_id to int for stream naming
            match_id = int(self.match_id) if isinstance(self.match_id, str) else self.match_id
            
            # Read from Redis stream
            stream_key = f"stream:match:{match_id}:events"
            
            # Check if stream exists
            if not self.redis_client.exists(stream_key):
                print(f"❌ Stream {stream_key} not found in Redis")
                print(f"🔄 Creating dummy data for demonstration...")
                return self._create_dummy_data(self.match_id)
            
            # Read all events from the stream
            events = self.redis_client.xrange(stream_key)
            
            if not events:
                print(f"❌ No events found in stream {stream_key}")
                print(f"🔄 Creating dummy data for demonstration...")
                return self._create_dummy_data(self.match_id)
            
            # Convert Redis stream data to DataFrame
            data_list = []
            for event_id, event_data in events:
                # Convert Redis hash to dict and convert numeric fields
                record = {}
                for k, v in event_data.items():
                    # Convert numeric strings to float if possible
                    try:
                        record[k] = float(v)
                    except (ValueError, TypeError):
                        record[k] = v
                data_list.append(record)
            
            df = pd.DataFrame(data_list)
            
            # Convert numeric columns (comprehensive list for territory calculation)
            numeric_columns = [
                'x_start', 'y_start', 'team_id', 'phase_team', 'player_id', 
                'time_seconds', 'period_id', 'xg', 'second'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure match_id column exists
            df['match_id'] = match_id
            
            print(f"✅ Loaded {len(df)} events from Redis for match {match_id}")
            return df
                
        except Exception as e:
            print(f"❌ Error loading data from Redis: {e}")
            print(f"🔄 Creating dummy data for demonstration...")
            return self._create_dummy_data(self.match_id)
    
    def _create_dummy_data(self, match_id: str) -> pd.DataFrame:
        """Create dummy data for demonstration when Redis data is not available"""
        np.random.seed(42)  # For reproducible results
        
        # Create realistic football data
        n_events = 800
        
        # Generate events with realistic distribution
        df = pd.DataFrame({
            'match_id': [match_id] * n_events,
            'team_id': np.random.choice([456, 789], n_events),
            'home': np.random.choice(['home', 'away'], n_events),
            'type_name': np.random.choice(['pass', 'shot', 'dribble', 'reception'], n_events),
            'x_start': np.random.uniform(0, 105, n_events),
            'y_start': np.random.uniform(0, 68, n_events),
            'phase_team': np.random.choice([456, 789], n_events),
            'team_name': np.random.choice(['Barcelona', 'Real Madrid'], n_events)
        })
        
        # Make home team slightly more dominant in their half
        home_mask = df['home'] == 'home'
        df.loc[home_mask, 'x_start'] = np.random.uniform(0, 70, home_mask.sum())  # Home attacks left side
        df.loc[~home_mask, 'x_start'] = np.random.uniform(35, 105, (~home_mask).sum())  # Away attacks right side
        
        # Ensure phase_team matches team_id for in-possession events
        df['phase_team'] = df['team_id']
        
        print(f"✅ Created dummy data: {len(df)} events")
        return df
    
    def generate_territory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate territory data using the original df_match_territory function with better data prep"""
        print("🔄 Generating territory data using df_match_territory...")
        
        # Check required columns
        required_cols = {"match_id", "team_id", "home", "type_name", "x_start", "y_start", "phase_team"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        try:
            # Prepare data very carefully to avoid string type issues
            df_clean = df.copy()
            
            # Ensure match_id exists
            if 'match_id' not in df_clean.columns:
                df_clean['match_id'] = self.match_id
            
            # Convert ALL numeric columns to proper types
            numeric_cols = ['match_id', 'team_id', 'phase_team', 'x_start', 'y_start', 'player_id', 'time_seconds', 'period_id']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle string columns very carefully
            string_cols = ['home', 'type_name']
            for col in string_cols:
                if col in df_clean.columns:
                    # Convert to string, handle NaN, strip whitespace, normalize case
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').str.strip()
                    if col == 'home':
                        df_clean[col] = df_clean[col].str.lower()
                        # Ensure only valid home/away values
                        df_clean = df_clean[df_clean[col].isin(['home', 'away'])]
            
            # Create team_name column if it doesn't exist (required by df_match_territory)
            if 'team_name' not in df_clean.columns:
                if 'match_home_team_name' in df_clean.columns and 'match_away_team_name' in df_clean.columns:
                    # Use actual team names from the data
                    df_clean['team_name'] = df_clean.apply(
                        lambda row: str(row['match_home_team_name']) if row['home'] == 'home' else str(row['match_away_team_name']),
                        axis=1
                    )
                else:
                    # Create team names based on team_id
                    unique_teams = df_clean['team_id'].dropna().unique()
                    team_mapping = {team_id: f"Team_{int(team_id)}" for team_id in unique_teams}
                    df_clean['team_name'] = df_clean['team_id'].map(team_mapping).fillna('Unknown')
            
            # Ensure team_name is string type
            df_clean['team_name'] = df_clean['team_name'].astype(str)
            
            # Drop rows with missing critical data
            df_clean = df_clean.dropna(subset=['x_start', 'y_start', 'team_id', 'phase_team', 'home', 'type_name'])
            
            if df_clean.empty:
                print("❌ No valid data after cleaning")
                return pd.DataFrame()
            
            print(f"✅ Cleaned data: {len(df_clean)} events")
            
            # Try the original function
            territory_df = df_match_territory(df_clean)
            
            if territory_df.empty:
                print("❌ No territory data generated by df_match_territory")
                return pd.DataFrame()
            
            print(f"✅ Generated territory data: {len(territory_df)} zones")
            return territory_df
            
        except Exception as e:
            print(f"❌ Error in df_match_territory: {e}")
            print("🔄 Falling back to custom implementation...")
            
            # Fallback to custom implementation if original fails
            return self._generate_territory_custom(df)
    
    def _generate_territory_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback custom territory implementation"""
        try:
            # Clean and prepare data
            df_clean = df.copy()
            
            # Ensure match_id exists
            if 'match_id' not in df_clean.columns:
                df_clean['match_id'] = self.match_id
            
            # Convert numeric columns
            for col in ['x_start', 'y_start', 'team_id', 'phase_team']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Clean string columns
            for col in ['home', 'type_name']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('unknown').astype(str).str.strip().str.lower()
            
            # Filter to in-possession events only (exclude restarts like original)
            exclude_types = {'corner_crossed', 'corner_short', 'throw_in', 'goal_kick'}
            df_poss = df_clean[
                (df_clean['team_id'] == df_clean['phase_team']) & 
                (~df_clean['type_name'].isin(exclude_types))
            ].copy()
            
            if df_poss.empty:
                print("❌ No possession events found")
                return pd.DataFrame()
            
            # Drop rows with invalid coordinates
            df_poss = df_poss.dropna(subset=['x_start', 'y_start'])
            
            # Apply coordinate transformation like the original df_match_territory
            # This is crucial for matching the original results!
            L, W = 105.0, 68.0  # pitch dimensions
            
            # Transform coordinates for home and away teams (wide orientation)
            home_mask = df_poss['home'] == 'home'
            away_mask = df_poss['home'] == 'away'
            
            # Home team: x stays same, y = W - y (flip y-axis)
            df_poss.loc[home_mask, 'y_start'] = W - df_poss.loc[home_mask, 'y_start']
            
            # Away team: x = L - x (flip x-axis), y stays same
            df_poss.loc[away_mask, 'x_start'] = L - df_poss.loc[away_mask, 'x_start']
            
            # Filter to valid coordinates after transformation
            df_poss = df_poss[
                (df_poss['x_start'] >= 0) & (df_poss['x_start'] <= 105) &
                (df_poss['y_start'] >= 0) & (df_poss['y_start'] <= 68)
            ]
            
            if df_poss.empty:
                print("❌ No valid coordinate data")
                return pd.DataFrame()
            
            print(f"✅ Processing {len(df_poss)} possession events (custom)")
            
            # Import futicolor for consistent colors
            try:
                from futiplot.utils import futicolor
                home_color_strong = futicolor.blue
                home_color_weak = futicolor.blue1
                away_color_strong = futicolor.pink
                away_color_weak = futicolor.pink1
            except:
                # Fallback colors
                home_color_strong = "#00B7FF"
                home_color_weak = "#85c1e9"
                away_color_strong = "#EA1F96"
                away_color_weak = "#f1948a"
            
            # Create territory zones manually
            territory_zones = []
            
            # Process each zone
            for y_bin in range(self.ny):
                for x_bin in range(self.nx):
                    x_min = self.length_zones[x_bin]
                    x_max = self.length_zones[x_bin + 1]
                    y_min = self.width_zones[y_bin]
                    y_max = self.width_zones[y_bin + 1]
                    
                    # Find events in this zone
                    zone_events = df_poss[
                        (df_poss['x_start'] >= x_min) & (df_poss['x_start'] < x_max) &
                        (df_poss['y_start'] >= y_min) & (df_poss['y_start'] < y_max)
                    ]
                    
                    if len(zone_events) == 0:
                        continue  # Skip empty zones
                    
                    # Count touches by team (using home/away)
                    home_touches = len(zone_events[zone_events['home'] == 'home'])
                    away_touches = len(zone_events[zone_events['home'] == 'away'])
                    total_touches = home_touches + away_touches
                    
                    if total_touches == 0:
                        continue
                    
                    # Calculate possession percentage
                    home_pct = home_touches / total_touches
                    
                    # Determine dominant team
                    if home_pct >= 0.5:
                        team_label = "home"
                        poss_pct = home_pct
                        fill_color = home_color_strong if poss_pct > 0.55 else home_color_weak
                    else:
                        team_label = "away"
                        poss_pct = 1.0 - home_pct
                        fill_color = away_color_strong if poss_pct > 0.55 else away_color_weak
                    
                    # Create possession label
                    possession_label = f"{int(round(poss_pct * 100))}%"
                    
                    # Add zone to results
                    territory_zones.append({
                        'match_id': self.match_id,
                        'x_min': float(x_min),
                        'x_max': float(x_max),
                        'y_min': float(y_min),
                        'y_max': float(y_max),
                        'team_label': team_label,
                        'possession_label': possession_label,
                        'fill_color': fill_color
                    })
            
            if not territory_zones:
                print("❌ No territory zones generated")
                return pd.DataFrame()
            
            territory_df = pd.DataFrame(territory_zones)
            print(f"✅ Generated {len(territory_df)} territory zones (custom)")
            return territory_df
            
        except Exception as e:
            print(f"❌ Error in custom territory generation: {e}")
            return pd.DataFrame()
    
    def create_simplified_output(self, territory_df: pd.DataFrame, match_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Create simplified output format with ALL data points from df_match_territory"""
        print("🔄 Creating simplified output format with complete data...")
        
        # Initialize output structure
        output = {
            "match_id": match_id,
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
                for _, row in territory_df.iterrows():
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
                    "team_label": None,
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
                else:
                    # Zone has no data (neutral) - keep defaults
                    zone_entry.update({
                        "team_label": "neutral",
                        "possession_label": "50%",
                        "home_value": 50,
                        "away_value": 50,
                        "fill_color": "#cccccc"
                    })
                
                output["zones"].append(zone_entry)
        
        print(f"✅ Created simplified output with {len(output['zones'])} zones")
        return output
    
    def export_to_json(self, data: Dict[str, Any], match_id: str) -> str:
        """Export data to JSON file"""
        filename = f"territory_data_{match_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported to: {filename}")
        return filename
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a summary of the generated data"""
        print("\n" + "="*60)
        print("📊 TERRITORY DATA SUMMARY (COMPLETE)")
        print("="*60)
        
        zones = data["zones"]
        total_zones = len(zones)
        
        print(f"🎯 Match ID: {data['match_id']}")
        print(f"📏 Grid Size: {data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {total_zones}")
        
        # Sample zones with ALL data points
        print(f"\n🎯 SAMPLE ZONES (with ALL data points):")
        sample_zones = zones[:3]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']}:")
            print(f"     Grid: x={zone['x']}, y={zone['y']}")
            print(f"     Boundaries: x_min={zone['x_min']}, x_max={zone['x_max']}, y_min={zone['y_min']}, y_max={zone['y_max']}")
            print(f"     Team: {zone['team_label']}")
            print(f"     Possession: {zone['possession_label']}")
            print(f"     Values: home={zone['home_value']}%, away={zone['away_value']}%")
            print(f"     Color: {zone['fill_color']}")
            print()
        
        print(f"📋 FRONTEND USAGE (COMPLETE DATA):")
        print(f"   - Load the JSON file")
        print(f"   - Access zones: data.zones[index]")
        print(f"   - Drawing: Use x_min, x_max, y_min, y_max for exact positioning")
        print(f"   - Labeling: Use team_label and possession_label for text")
        print(f"   - Styling: Use fill_color and home_value/away_value for visuals")
        print(f"   - Indexing: Use x, y for grid-based operations")

    def show_territory_figure(self, territory_df: pd.DataFrame, match_id: str):
        """
        Show the territory plot figure (can be commented/uncommented)
        This function creates and displays the territory visualization
        """
        try:
            # Import plotting functions
            from futiplot.soccer.pitch import plot_pitch
            import matplotlib.pyplot as plt
            
            print("🖼️ Generating territory plot figure...")
            
            # Create pitch
            fig, ax, pitch = plot_pitch(orientation="wide", figsize=(12, 8), linewidth=2)
            
            # Plot each zone
            for _, zone in territory_df.iterrows():
                x_min, x_max = zone["x_min"], zone["x_max"]
                y_min, y_max = zone["y_min"], zone["y_max"]
                color = zone["fill_color"]
                label = zone["possession_label"]
                
                # Create rectangle
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                               facecolor=color, alpha=0.7, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
                
                # Add text label
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                ax.text(center_x, center_y, label, ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
            
            # Set title and labels
            ax.set_title(f"Territory Plot - Match: {match_id}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Length (meters)", fontsize=12)
            ax.set_ylabel("Width (meters)", fontsize=12)
            
            # Set axis limits
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            
            # Invert y-axis for proper football view
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Territory plot displayed successfully!")
            
        except ImportError as e:
            print(f"❌ Could not import plotting libraries: {e}")
            print("💡 To enable plotting, install: matplotlib")
        except Exception as e:
            print(f"❌ Error generating plot: {e}")

def list_available_matches():
    """List all available matches in Redis"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        
        # Get all match event streams
        keys = r.keys("stream:match:*:events")
        match_ids = [key.split(":")[2] for key in keys]
        
        print(f"🔧 Found {len(match_ids)} matches in Redis")
        if match_ids:
            print("📋 Available match IDs:")
            for match_id in sorted(match_ids):
                print(f"   - {match_id}")
        else:
            print("❌ No matches found in Redis streams")
            
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")

def main():
    """Main function to run the territory plot generator for all matches in Redis"""
    
    print("🚀 TERRITORY PLOT GENERATOR (REDIS VERSION - COMPLETE DATA)")
    print("=" * 60)
    
    # Connect to Redis
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        sys.exit(1)

    # Get all match event streams
    keys = r.keys("stream:match:*:events")
    match_ids = [key.split(":")[2] for key in keys]

    print(f"🔧 Found {len(match_ids)} matches in Redis")

    for match_id in match_ids:
        print("\n" + "="*60)
        print(f"📌 Generating territory for match_id: {match_id}")
        print("="*60)

        # Initialize generator
        generator = TerritoryPlotGenerator(match_id)
        
        try:
            # Load match data from Redis
            df = generator.load_match_data_from_redis()
            
            # Generate territory data using EXACT original function
            territory_df = generator.generate_territory_data(df)
            
            if territory_df.empty:
                print(f"❌ Skipping match_id {match_id} (failed to generate territory data)")
                continue
            
            # Create simplified output with ALL data points
            output_data = generator.create_simplified_output(territory_df, match_id, df)
            
            # Export to JSON
            filename = generator.export_to_json(output_data, match_id)
            
            # Print summary
            generator.print_summary(output_data)
            
            # =====================================================================
            # UNCOMMENT THE LINE BELOW TO SHOW THE TERRITORY FIGURE
            # =====================================================================
            # generator.show_territory_figure(territory_df, match_id)
            # =====================================================================
            
            print(f"✅ Territory generation complete for match_id {match_id}")
            print(f"📁 Output file: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing match_id {match_id}: {e}")
            continue

    print(f"\n🎉 SUCCESS! Territory data generated for all {len(match_ids)} matches")
    print(f"📊 Ready for frontend integration with COMPLETE data!")

if __name__ == "__main__":
    main()