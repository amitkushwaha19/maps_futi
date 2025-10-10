#!/usr/bin/env python3
"""
Action Heatmap Generator - Redis Version (Legacy Format)
A dedicated script for generating action heatmap data matching the 4491 format from Redis.

Usage:
    python action_heatmap_generator_redis_legacy.py
    
Processes all matches found in Redis automatically using legacy format.
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

try:
    from futiplot.match_card.match_card_charts import df_match_action_heatmap
    from futiplot.utils import load_sample_data
    print("✅ Successfully imported df_match_action_heatmap")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory and futiplot is available")
    sys.exit(1)

class ActionHeatmapGeneratorLegacy:
    """Generate action heatmap data for a specific match from Redis using legacy format"""
    
    def __init__(self, match_id):
        self.match_id = match_id
        self.redis_client = None
        
        # Grid dimensions (matching 4491 format exactly)
        self.x_zones = 21  # 20 cells along length (0,5,10,15...100)
        self.y_zones = 14  # 13 cells along width 
        self.total_zones = 20 * 13  # 260 zones
        
        # Pitch dimensions
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
        # 4491 grid parameters (reverse engineered)
        self.x_cell_size = 5.0  # Each cell is 5.0 units wide
        self.y_cell_size = 4.857142925262451  # Each cell is ~4.857 units tall
        
    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
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
            
            # Convert numeric columns
            numeric_columns = [
                'x_start', 'y_start', 'team_id', 'player_id', 
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
    
    def generate_heatmap_data_legacy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate heatmap data using the original df_match_action_heatmap function"""
        print("🔄 Generating action heatmap data using df_match_action_heatmap...")
        
        # Check required columns
        required_cols = {"match_id", "team_id", "home", "type_name", "x_start", "y_start"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        try:
            # Prepare data very carefully to avoid type issues
            df_clean = df.copy()
            
            # Ensure match_id exists
            if 'match_id' not in df_clean.columns:
                df_clean['match_id'] = self.match_id
            
            # Convert ALL numeric columns to proper types
            numeric_cols = ['match_id', 'team_id', 'x_start', 'y_start', 'player_id', 'time_seconds', 'period_id']
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
            
            print(f"✅ Generated heatmap data: {len(heatmap_df)} zones using original function")
            return heatmap_df
            
        except Exception as e:
            print(f"❌ Error in df_match_action_heatmap: {e}")
            print("❌ Cannot fallback for legacy format - original function required")
            return pd.DataFrame()
    
    def create_simplified_output_legacy(self, heatmap_df: pd.DataFrame, match_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Create simplified output format matching exact 4491 structure"""
        print("🔄 Creating simplified output format (4491 format)...")
        
        # Get unique team IDs from original data (StatsBomb team IDs)
        unique_teams = []
        if not original_df.empty and 'team_id' in original_df.columns:
            unique_teams = [str(int(tid)) for tid in original_df['team_id'].dropna().unique()]
        
        # Initialize output structure (EXACT 4491 FORMAT)
        output = {
            "match_id": match_id,
            "pitch_info": {
                "length_meters": self.pitch_length,
                "width_meters": self.pitch_width,
                "grid_size": "20 x 13",  # Exact 4491 format
                "total_zones": 260,
                "teams": unique_teams
            },
            "zones": []
        }
        
        # Create 4491-style grid manually (20x13 = 260 zones)
        # X: 0,5,10,15...100 (21 edges = 20 cells)
        # Y: 0,4.857,9.714...62.285 (14 edges = 13 cells)
        x_edges = [i * 5.0 for i in range(21)]  # 0, 5, 10, 15, ..., 100
        y_edges = [i * self.y_cell_size for i in range(14)]  # 0, 4.857, 9.714, ..., 62.285
        
        # Calculate all events statistics for proper normalization
        all_counts = []
        for team_id in unique_teams:
            team_events = original_df[original_df['team_id'] == float(team_id)]
            for i in range(20):  # x cells
                for j in range(13):  # y cells
                    x_min, x_max = x_edges[i], x_edges[i + 1]
                    y_min, y_max = y_edges[j], y_edges[j + 1]
                    
                    zone_events = team_events[
                        (team_events['x_start'] >= x_min) & 
                        (team_events['x_start'] < x_max) &
                        (team_events['y_start'] >= y_min) & 
                        (team_events['y_start'] < y_max)
                    ]
                    all_counts.append(len(zone_events))
        
        # Calculate normalization parameters (like 4491)
        max_count = max(all_counts) if all_counts else 1
        mean_count = np.mean([c for c in all_counts if c > 0]) if any(c > 0 for c in all_counts) else 1
        
        # Debug: Check coordinate ranges in the data
        print(f"🔍 Data coordinate ranges:")
        print(f"   X range: {original_df['x_start'].min():.2f} to {original_df['x_start'].max():.2f}")
        print(f"   Y range: {original_df['y_start'].min():.2f} to {original_df['y_start'].max():.2f}")
        
        # Debug: Check if we have events in (0,0) zone
        zone_00_events = original_df[
            (original_df['x_start'] >= 0) & (original_df['x_start'] < 5) &
            (original_df['y_start'] >= 0) & (original_df['y_start'] < 4.857)
        ]
        print(f"   Events in (0.0,0.0) zone: {len(zone_00_events)}")
        
        # Generate zones for each team
        for team_id in unique_teams:
            team_events = original_df[original_df['team_id'] == float(team_id)]
            print(f"🔍 Team {team_id}: {len(team_events)} events")
            
            for i in range(20):  # x cells (0 to 19)
                for j in range(13):  # y cells (0 to 12)
                    x_min, x_max = x_edges[i], x_edges[i + 1]
                    y_min, y_max = y_edges[j], y_edges[j + 1]
                    
                    # Find events in this zone for this team
                    zone_events = team_events[
                        (team_events['x_start'] >= x_min) & 
                        (team_events['x_start'] < x_max) &
                        (team_events['y_start'] >= y_min) & 
                        (team_events['y_start'] < y_max)
                    ]
                    
                    event_count = len(zone_events)
                    
                    # DEBUG: Print first few zones to see what's happening
                    if i < 3 and j < 3:
                        print(f"   Zone ({x_min},{y_min}) team {team_id}: {event_count} events")
                    
                    # Include ALL zones (even with 0 events) to match 4491 exactly
                    # Calculate intensity and alpha using 4491-style normalization
                    if event_count > 0:
                        intensity = 1.0 + (event_count / mean_count) * 0.8
                        alpha = min(1.0, event_count / (max_count * 0.7))
                    else:
                        # For zones with 0 events, use minimal values
                        intensity = 1.0
                        alpha = 0.0
                    
                    # Only include zones with events (like 4491 does)
                    if event_count > 0:
                        zone_entry = {
                            "zone_id": f"({x_min},{y_min})",
                            "team_id": f"{team_id}.0",
                            "x_min": float(x_min),
                            "x_max": float(x_max),
                            "y_min": float(y_min),
                            "y_max": float(y_max),
                            "count": event_count,
                            "intensity": round(intensity, 15),
                            "alpha": round(alpha, 17),
                            "center_x": float((x_min + x_max) / 2),
                            "center_y": float((y_min + y_max) / 2)
                        }
                        output["zones"].append(zone_entry)
        
        print(f"✅ Created 4491-format output with {len(output['zones'])} zones")
        return output
    
    def export_to_json(self, data: Dict[str, Any], match_id: str) -> str:
        """Export data to JSON file"""
        filename = f"action_heatmap_legacy_{match_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported to: {filename}")
        return filename
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a summary of the generated data"""
        print("\n" + "="*60)
        print("📊 ACTION HEATMAP DATA SUMMARY (4491 FORMAT)")
        print("="*60)
        
        zones = data["zones"]  # Changed from heatmap_data to zones
        total_zones = len(zones)
        
        print(f"🎯 Match ID: {data['match_id']}")
        print(f"📏 Grid Size: {data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {total_zones}")
        print(f"👥 Teams: {data['pitch_info']['teams']}")
        
        # Statistics
        total_events = [z["count"] for z in zones]
        team_counts = {}
        for zone in zones:
            team_id = zone["team_id"]
            if team_id not in team_counts:
                team_counts[team_id] = 0
            team_counts[team_id] += zone["count"]
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total events (all zones): {sum(total_events)}")
        for team_id, count in team_counts.items():
            print(f"   Team {team_id}: {count} events")
        
        # Sample zones
        print(f"\n🎯 SAMPLE ZONES (team_id, count, intensity, alpha):")
        sample_zones = zones[:5]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']}: Team={zone['team_id']}, Count={zone['count']}, Intensity={zone['intensity']:.3f}, Alpha={zone['alpha']:.3f}")
        
        print(f"\n📋 FRONTEND USAGE (4491 FORMAT):")
        print(f"   - Load the JSON file")
        print(f"   - Access zones: data.zones[index]")
        print(f"   - Get coordinates: zone.x_min, zone.x_max, zone.y_min, zone.y_max")
        print(f"   - Get team data: zone.team_id, zone.count")
        print(f"   - Get visualization: zone.intensity, zone.alpha")

def main():
    """Main function to run the legacy action heatmap generator for all matches in Redis"""
    
    print("🚀 ACTION HEATMAP GENERATOR (REDIS VERSION - LEGACY FORMAT)")
    print("=" * 70)
    
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
        print("\n" + "="*70)
        print(f"📌 Generating legacy action heatmap for match_id: {match_id}")
        print("="*70)

        # Initialize generator
        generator = ActionHeatmapGeneratorLegacy(match_id)
        
        try:
            # Load match data from Redis
            df = generator.load_match_data_from_redis()
            
            if df.empty:
                print(f"❌ Skipping match_id {match_id} (no data)")
                continue
            
            # Generate heatmap data using legacy format
            heatmap_df = generator.generate_heatmap_data_legacy(df)
            
            if heatmap_df.empty:
                print(f"❌ Skipping match_id {match_id} (failed to generate heatmap data)")
                continue
            
            # Create legacy output format
            output_data = generator.create_simplified_output_legacy(heatmap_df, match_id, df)
            
            # Export to JSON
            filename = generator.export_to_json(output_data, match_id)
            
            # Print summary
            generator.print_summary(output_data)
            
            print(f"✅ Legacy action heatmap generation complete for match_id {match_id}")
            print(f"📁 Output file: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing match_id {match_id}: {e}")
            continue

    print(f"\n🎉 SUCCESS! Legacy action heatmap data generated for all {len(match_ids)} matches")
    print(f"📊 Ready for frontend integration with legacy format!")

if __name__ == "__main__":
    main()