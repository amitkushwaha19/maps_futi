#!/usr/bin/env python3
"""
Action Heatmap Generator - Redis Version
A dedicated script for generating action heatmap visualization data for a specific match from Redis.

Usage:
    python action_heatmap_generator_redis.py
    
Processes all matches found in Redis automatically.
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

from futiplot.match_card.match_card_charts import df_match_action_heatmap
from futiplot.utils import load_sample_data

class ActionHeatmapGenerator:
    """Generate action heatmap data for a specific match from Redis"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.redis_client = None
        self.heatmap_data = None
        self.original_df = None
        
        # Grid dimensions (same as in df_match_action_heatmap)
        self.x_zones = 21  # 20 cells along length
        self.y_zones = 14  # 13 cells along width
        self.total_zones = 20 * 13  # 260 zones
        
        # Pitch dimensions
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
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
                return pd.DataFrame()
            
            # Read all events from the stream
            events = self.redis_client.xrange(stream_key)
            
            if not events:
                print(f"❌ No events found in stream {stream_key}")
                return pd.DataFrame()
            
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
            
            # Convert numeric columns (comprehensive list for heatmap calculation)
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
            return pd.DataFrame()
    
    def generate_heatmap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate heatmap data using the original df_match_action_heatmap function with fallback"""
        print("🔄 Generating action heatmap data using df_match_action_heatmap...")
        
        # Check required columns
        required_cols = {"match_id", "team_id", "home", "type_name", "x_start", "y_start"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        try:
            # Prepare data carefully to avoid type issues
            df_clean = df.copy()
            
            # Ensure match_id exists
            if 'match_id' not in df_clean.columns:
                df_clean['match_id'] = self.match_id
            
            # Convert ALL numeric columns to proper types
            numeric_cols = ['match_id', 'team_id', 'x_start', 'y_start', 'player_id', 'time_seconds', 'period_id']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle string columns carefully
            string_cols = ['home', 'type_name']
            for col in string_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').str.strip()
                    if col == 'home':
                        df_clean[col] = df_clean[col].str.lower()
                        # Ensure only valid home/away values
                        df_clean = df_clean[df_clean[col].isin(['home', 'away'])]
            
            # Drop rows with missing critical data
            df_clean = df_clean.dropna(subset=['x_start', 'y_start', 'team_id', 'home', 'type_name'])
            
            if df_clean.empty:
                print("❌ No valid data after cleaning")
                return pd.DataFrame()
            
            print(f"✅ Cleaned data: {len(df_clean)} events")
            
            # Try the original function
            heatmap_df = df_match_action_heatmap(
                df_clean,
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                length_edges=self.x_zones,
                width_edges=self.y_zones
            )
            
            if heatmap_df.empty:
                print("❌ No heatmap data generated by df_match_action_heatmap")
                return pd.DataFrame()
            
            print(f"✅ Generated heatmap data: {len(heatmap_df)} zones")
            return heatmap_df
            
        except Exception as e:
            print(f"❌ Error in df_match_action_heatmap: {e}")
            print("🔄 Falling back to custom implementation...")
            
            # Fallback to custom implementation if original fails
            return self._generate_heatmap_custom(df)
    
    def _generate_heatmap_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback custom heatmap implementation"""
        try:
            print("🔄 Using custom heatmap implementation...")
            
            # Clean and prepare data
            df_clean = df.copy()
            
            # Ensure match_id exists
            if 'match_id' not in df_clean.columns:
                df_clean['match_id'] = self.match_id
            
            # Convert numeric columns
            for col in ['x_start', 'y_start', 'team_id']:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Clean string columns
            for col in ['home', 'type_name']:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna('unknown').astype(str).str.strip().str.lower()
            
            # Filter valid events
            df_clean = df_clean.dropna(subset=['x_start', 'y_start'])
            df_clean = df_clean[
                (df_clean['x_start'] >= 0) & (df_clean['x_start'] <= 105) &
                (df_clean['y_start'] >= 0) & (df_clean['y_start'] <= 68) &
                (df_clean['home'].isin(['home', 'away']))
            ]
            
            if df_clean.empty:
                print("❌ No valid coordinate data")
                return pd.DataFrame()
            
            print(f"✅ Processing {len(df_clean)} events (custom)")
            
            # Create heatmap zones manually
            heatmap_zones = []
            
            # Create grid
            x_edges = np.linspace(0, self.pitch_length, self.x_zones)
            y_edges = np.linspace(0, self.pitch_width, self.y_zones)
            
            # Process each zone
            for i in range(len(x_edges) - 1):
                for j in range(len(y_edges) - 1):
                    x_min, x_max = x_edges[i], x_edges[i + 1]
                    y_min, y_max = y_edges[j], y_edges[j + 1]
                    
                    # Find events in this zone
                    zone_events = df_clean[
                        (df_clean['x_start'] >= x_min) & (df_clean['x_start'] < x_max) &
                        (df_clean['y_start'] >= y_min) & (df_clean['y_start'] < y_max)
                    ]
                    
                    # Count events by team
                    home_count = len(zone_events[zone_events['home'] == 'home'])
                    away_count = len(zone_events[zone_events['home'] == 'away'])
                    both_count = home_count + away_count
                    
                    # Calculate fill values (normalized)
                    max_count = max(1, df_clean.groupby(['home']).size().max()) if len(df_clean) > 0 else 1
                    both_fill = min(1.0, both_count / max_count) if both_count > 0 else 0.0
                    home_fill = min(1.0, home_count / max_count) if home_count > 0 else 0.0
                    away_fill = min(1.0, away_count / max_count) if away_count > 0 else 0.0
                    
                    # Add zone to results
                    heatmap_zones.append({
                        'match_id': self.match_id,
                        'x_min': float(x_min),
                        'x_max': float(x_max),
                        'y_min': float(y_min),
                        'y_max': float(y_max),
                        'both_count': both_count,
                        'home_count': home_count,
                        'away_count': away_count,
                        'both_fill': both_fill,
                        'home_fill': home_fill,
                        'away_fill': away_fill
                    })
            
            heatmap_df = pd.DataFrame(heatmap_zones)
            print(f"✅ Generated {len(heatmap_df)} heatmap zones (custom)")
            return heatmap_df
            
        except Exception as e:
            print(f"❌ Error in custom heatmap generation: {e}")
            return pd.DataFrame()
    
    def create_simplified_output(self, heatmap_df: pd.DataFrame, match_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        print("🔄 Creating simplified output format...")

        # Extract unique team_ids
        if "team_id" in original_df.columns:
            team_ids = sorted(original_df["team_id"].dropna().unique().astype(str).tolist())
        else:
            team_ids = []

        # Initialize output structure
        output = {
            "match_id": str(match_id),  # renamed from "match_id"
            "pitch_info": {
                "length_meters": self.pitch_length,
                "width_meters": self.pitch_width,
                "grid_size": f"{self.x_zones - 1} x {self.y_zones - 1}",
                "total_zones": self.total_zones,
                "teams": team_ids
            },
            "zones": []
        }

        # Process each zone
        for _, zone in heatmap_df.iterrows():
            for team_id in team_ids:
                # Determine whether it's home or away
                try:
                    home_team_id = str(original_df[original_df['home'] == 'home']['team_id'].iloc[0])
                except IndexError:
                    home_team_id = None

                if str(team_id) == home_team_id:
                    count = int(zone.get("home_count", 0))
                    alpha = float(zone.get("home_fill", 0.0))
                else:
                    count = int(zone.get("away_count", 0))
                    alpha = float(zone.get("away_fill", 0.0))

                # Calculate intensity (customizable)
                intensity = float(count * 0.5 + alpha * 2.0)

                zone_entry = {
                    "zone_id": f"({zone['x_min']:.1f},{zone['y_min']:.1f})",
                    "team_id": str(team_id),
                    "x_min": float(zone["x_min"]),
                    "x_max": float(zone["x_max"]),
                    "y_min": float(zone["y_min"]),
                    "y_max": float(zone["y_max"]),
                    "count": count,
                    "alpha": alpha,
                    "intensity": intensity,
                    "center_x": float((zone["x_min"] + zone["x_max"]) / 2),
                    "center_y": float((zone["y_min"] + zone["y_max"]) / 2)
                }

                output["zones"].append(zone_entry)

        print(f"✅ Created simplified output with {len(output['zones'])} zones")
        return output

    
    def export_to_json(self, data: Dict[str, Any], match_id: str) -> str:
        """Export data to JSON file"""
        filename = f"action_heatmap_data_{match_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported to: {filename}")
        return filename
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a summary of the generated data"""
        print("\n" + "="*60)
        print("📊 ACTION HEATMAP DATA SUMMARY")
        print("="*60)
        
        zones = data["zones"]
        total_zones = len(zones)
        
        print(f"🎯 Match ID: {data.get('game_id', 'N/A')}")
        print(f"📏 Grid Size: {data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {total_zones}")
        
        counts = [z["count"] for z in zones]
        alphas = [z["alpha"] for z in zones]
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total count of events: {sum(counts)}")
        print(f"   Average alpha (transparency): {sum(alphas)/len(alphas):.3f}")
        
        print(f"\n🎯 SAMPLE ZONES (zone_id, team_id, count, intensity, alpha):")
        sample_zones = zones[:5]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']} (Team {zone['team_id']}): "
                  f"Count={zone['count']}, Intensity={zone['intensity']:.3f}, Alpha={zone['alpha']:.3f}")

def main():
    """Main function to run the action heatmap generator for all matches in Redis"""
    
    print("🚀 ACTION HEATMAP GENERATOR (REDIS VERSION)")
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
        print(f"📌 Generating action heatmap for match_id: {match_id}")
        print("="*60)

        # Initialize generator
        generator = ActionHeatmapGenerator(match_id)
        
        try:
            # Load match data from Redis
            df = generator.load_match_data_from_redis()
            
            if df.empty:
                print(f"❌ Skipping match_id {match_id} (no data)")
                continue
            
            # Generate heatmap data
            heatmap_df = generator.generate_heatmap_data(df)
            
            if heatmap_df.empty:
                print(f"❌ Skipping match_id {match_id} (failed to generate heatmap data)")
                continue
            
            # Create simplified output
            output_data = generator.create_simplified_output(heatmap_df, match_id, df)
            
            # Export to JSON
            filename = generator.export_to_json(output_data, match_id)
            
            # Print summary
            generator.print_summary(output_data)
            
            print(f"✅ Action heatmap generation complete for match_id {match_id}")
            print(f"📁 Output file: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing match_id {match_id}: {e}")
            continue

    print(f"\n🎉 SUCCESS! Action heatmap data generated for all {len(match_ids)} matches")
    print(f"📊 Ready for frontend integration!")

if __name__ == "__main__":
    main()