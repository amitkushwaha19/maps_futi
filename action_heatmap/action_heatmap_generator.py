#!/usr/bin/env python3
"""
Action Heatmap Generator - FIXED VERSION
A dedicated script for generating action heatmap visualization data for a specific match.

Usage:
    python action_heatmap_generator_fixed.py <game_id>
    
Example:
    python action_heatmap_generator_fixed.py 4491
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from futiplot.match_card.match_card_charts import df_match_action_heatmap
from futiplot.utils import load_sample_data

class ActionHeatmapGenerator:
    """Generate action heatmap data for a specific match"""
    
    def __init__(self):
        # Grid dimensions (same as in df_match_action_heatmap)
        self.x_zones = 21  # 20 cells along length
        self.y_zones = 14  # 13 cells along width
        self.total_zones = 20 * 13  # 260 zones
        
        # Pitch dimensions
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        
    def load_match_data(self, game_id: str) -> pd.DataFrame:
        """Load match data for a specific game_id"""
        try:
            print(f"🔄 Loading data for game: {game_id}")
            df = load_sample_data("vaep")
            
            # Try to find the game_id (handle both string and integer types)
            game_id_int = None
            try:
                game_id_int = int(game_id)
            except ValueError:
                pass
            
            # Check if game_id exists (as string or integer)
            if game_id in df["game_id"].values or (game_id_int is not None and game_id_int in df["game_id"].values):
                # Use the actual game_id from the data
                actual_game_id = game_id_int if game_id_int is not None and game_id_int in df["game_id"].values else game_id
                match_data = df[df["game_id"] == actual_game_id].copy()
                print(f"✅ Found {len(match_data)} events for game {actual_game_id}")
                return match_data
            else:
                print(f"❌ Game {game_id} not found in sample data")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
            return pd.DataFrame()
    
    def generate_heatmap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate heatmap data using the UPDATED function signature"""
        print("🔄 Generating action heatmap data using df_match_action_heatmap...")
        # FIXED: Updated to use new function signature
        heatmap_df = df_match_action_heatmap(
            df,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            length_edges=self.x_zones,
            width_edges=self.y_zones
        )
        
        if heatmap_df.empty:
            print("❌ No heatmap data generated")
            return pd.DataFrame()
        
        print(f"✅ Generated heatmap data: {len(heatmap_df)} zones")
        return heatmap_df
    
    def create_simplified_output(self, heatmap_df: pd.DataFrame, game_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Create simplified output format for frontend team - FIXED VERSION"""
        print("🔄 Creating simplified output format...")
        
        # Initialize output structure
        output = {
            "game_id": game_id,
            "pitch_info": {
                "length_meters": self.pitch_length,
                "width_meters": self.pitch_width,
                "grid_size": f"{self.x_zones-1} x {self.y_zones-1}",
                "total_zones": self.total_zones
            },
            "zones": []
        }
        
        # Process each zone (FIXED: Updated for new column structure)
        for _, zone in heatmap_df.iterrows():
            zone_entry = {
                "zone_id": f"({zone['x_min']:.1f},{zone['y_min']:.1f})",
                "x_min": float(zone["x_min"]),
                "x_max": float(zone["x_max"]),
                "y_min": float(zone["y_min"]),
                "y_max": float(zone["y_max"]),
                # NEW: Updated to use new column names
                "both_count": int(zone["both_count"]),
                "home_count": int(zone["home_count"]),
                "away_count": int(zone["away_count"]),
                "both_fill": float(zone["both_fill"]),
                "home_fill": float(zone["home_fill"]),
                "away_fill": float(zone["away_fill"]),
                "center_x": float((zone["x_min"] + zone["x_max"]) / 2),
                "center_y": float((zone["y_min"] + zone["y_max"]) / 2)
            }
            output["zones"].append(zone_entry)
        
        print(f"✅ Created simplified output with {len(output['zones'])} zones")
        return output
    
    def export_to_json(self, data: Dict[str, Any], game_id: str) -> str:
        """Export data to JSON file"""
        filename = f"action_heatmap_data_{game_id}.json"
        
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
        
        print(f"🎯 Game ID: {data['game_id']}")
        print(f"📏 Grid Size: {data['pitch_info']['grid_size']}")
        print(f"🏟️ Total Zones: {total_zones}")
        
        # Statistics
        both_counts = [z["both_count"] for z in zones]
        home_counts = [z["home_count"] for z in zones]
        away_counts = [z["away_count"] for z in zones]
        
        print(f"\n📊 STATISTICS:")
        print(f"   Total events (both): {sum(both_counts)}")
        print(f"   Total events (home): {sum(home_counts)}")
        print(f"   Total events (away): {sum(away_counts)}")
        
        # Sample zones
        print(f"\n🎯 SAMPLE ZONES (both_count, home_count, away_count, both_fill):")
        sample_zones = zones[:5]
        for zone in sample_zones:
            print(f"   Zone {zone['zone_id']}: Both={zone['both_count']}, Home={zone['home_count']}, Away={zone['away_count']}, Fill={zone['both_fill']:.3f}")
        
        print(f"\n📋 FRONTEND USAGE:")
        print(f"   - Load the JSON file")
        print(f"   - Access zones: data.zones[index]")
        print(f"   - Get values: zone.both_count, zone.home_count, zone.away_count")
        print(f"   - Get alpha: zone.both_fill, zone.home_fill, zone.away_fill")
        print(f"   - Draw rectangles using zone coordinates with alpha transparency")

def main():
    """Main function to run the action heatmap generator"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("❌ Usage: python action_heatmap_generator_fixed.py <game_id>")
        print("Example: python action_heatmap_generator_fixed.py 4491")
        sys.exit(1)
    
    game_id = sys.argv[1]
    
    print("🚀 ACTION HEATMAP GENERATOR (FIXED)")
    print("=" * 50)
    print(f"🎯 Target Game: {game_id}")
    
    # Initialize generator
    generator = ActionHeatmapGenerator()
    
    try:
        # Load match data
        df = generator.load_match_data(game_id)
        
        if df.empty:
            print("❌ No data found for this game")
            sys.exit(1)
        
        # Generate heatmap data using UPDATED function
        heatmap_df = generator.generate_heatmap_data(df)
        
        if heatmap_df.empty:
            print("❌ Failed to generate heatmap data")
            sys.exit(1)
        
        # Create simplified output for frontend
        output_data = generator.create_simplified_output(heatmap_df, game_id, df)
        
        # Export to JSON
        filename = generator.export_to_json(output_data, game_id)
        
        # Print summary
        generator.print_summary(output_data)
        
        print(f"\n🎉 SUCCESS! Action heatmap data generated for game {game_id}")
        print(f"📁 Output file: {filename}")
        print(f"📊 Ready for frontend integration!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
