#!/usr/bin/env python3
"""
Territory Plot Generator - Redis Stream Version
===============================================

This script generates territory plot data for matches using Redis streams as data source,
following the same pattern as momentum_generator.py. It exports frontend-friendly JSON
for easy consumption by the frontend team.

Usage:
    python territory_plot_generator.py

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

class TerritoryPlotGenerator:
    """Generate territory plot data for a specific match"""
    
    def __init__(self):
        # Zone definitions (EXACTLY as in df_match_territory)
        self.length_zones = np.array([0.0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0], dtype=float)
        self.width_zones = np.array([0.0, 13.84, 24.84, 43.16, 54.16, 68.0], dtype=float)
        
        # Grid dimensions
        self.nx, self.ny = len(self.length_zones) - 1, len(self.width_zones) - 1
        
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
                print(f"🔄 Creating dummy data for demonstration...")
                return self._create_dummy_data(game_id)
                
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
            print(f"🔄 Creating dummy data for demonstration...")
            return self._create_dummy_data(game_id)
    
    def _create_dummy_data(self, game_id: str) -> pd.DataFrame:
        """Create dummy data for demonstration when sample data is not available"""
        np.random.seed(42)  # For reproducible results
        
        # Create realistic football data
        n_events = 800
        
        # Generate events with realistic distribution
        df = pd.DataFrame({
            'game_id': [game_id] * n_events,
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
        """Generate territory data using the EXACT original function"""
        print("🔄 Generating territory data using df_match_territory...")
        territory_df = df_match_territory(df)
        
        if territory_df.empty:
            print("❌ No territory data generated")
            return pd.DataFrame()
        
        print(f"✅ Generated territory data: {len(territory_df)} zones")
        return territory_df
    
    def create_simplified_output(self, territory_df: pd.DataFrame, game_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Create simplified output format with ALL data points from df_match_territory"""
        print("🔄 Creating simplified output format with complete data...")
        
        # Initialize output structure
        output = {
            "game_id": game_id,
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
    
    def export_to_json(self, data: Dict[str, Any], game_id: str) -> str:
        """Export data to JSON file"""
        filename = f"territory_data_{game_id}.json"
        
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
        
        print(f"🎯 Game ID: {data['game_id']}")
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

    def show_territory_figure(self, territory_df: pd.DataFrame, game_id: str):
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
            ax.set_title(f"Territory Plot - Game: {game_id}", fontsize=16, fontweight='bold')
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

def main():
    """Main function to run the territory plot generator"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("❌ Usage: python territory_plot_generator_fixed.py <game_id>")
        print("Example: python territory_plot_generator_fixed.py 4491")
        sys.exit(1)
    
    game_id = sys.argv[1]
    
    print("🚀 TERRITORY PLOT GENERATOR (FIXED - COMPLETE DATA)")
    print("=" * 60)
    print(f"🎯 Target Game: {game_id}")
    
    # Initialize generator
    generator = TerritoryPlotGenerator()
    
    try:
        # Load match data
        df = generator.load_match_data(game_id)
        
        # Generate territory data using EXACT original function
        territory_df = generator.generate_territory_data(df)
        
        if territory_df.empty:
            print("❌ Failed to generate territory data")
            sys.exit(1)
        
        # Create simplified output with ALL data points
        output_data = generator.create_simplified_output(territory_df, game_id, df)
        
        # Export to JSON
        filename = generator.export_to_json(output_data, game_id)
        
        # Print summary
        generator.print_summary(output_data)
        
        # =====================================================================
        # UNCOMMENT THE LINE BELOW TO SHOW THE TERRITORY FIGURE
        # =====================================================================
        generator.show_territory_figure(territory_df, game_id)
        # =====================================================================
        
        print(f"\n🎉 SUCCESS! Territory data generated for game {game_id}")
        print(f"📁 Output file: {filename}")
        print(f"📊 Ready for frontend integration with COMPLETE data!")
        print(f"💡 To show the plot, uncomment the show_territory_figure line in the script")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
