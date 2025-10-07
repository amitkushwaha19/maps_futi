#!/usr/bin/env python3
"""
Action Heatmap Generator - Simplified Format
A dedicated script for generating action heatmap data in a simple x,y,value format.

Usage:
    python action_heatmap_generator_simple.py <game_id>
    
Example:
    python action_heatmap_generator_simple.py 4491
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add the src directory to path
sys.path.append("src")

from futiplot.match_card.match_card_charts import df_match_action_heatmap
from futiplot.utils import load_sample_data

class SimpleActionHeatmapGenerator:
    """Generate simplified action heatmap data for a specific match"""
    
    def __init__(self):
        # Grid dimensions (same as in df_match_action_heatmap)
        self.x_zones = 21  # 20 cells along length
        self.y_zones = 14  # 13 cells along width
        
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
        n_events = 1200
        
        # Generate events with realistic distribution
        df = pd.DataFrame({
            'game_id': [game_id] * n_events,
            'team_id': np.random.choice([456, 789], n_events),
            'home': np.random.choice(['home', 'away'], n_events),
            'type_name': np.random.choice(['pass', 'shot', 'dribble', 'reception', 'tackle', 'interception'], n_events),
            'x_start': np.random.uniform(0, 105, n_events),
            'y_start': np.random.uniform(0, 68, n_events),
            'player_id': np.random.randint(1000, 9999, n_events)
        })
        
        # Make home team more active in their half
        home_mask = df['home'] == 'home'
        df.loc[home_mask, 'x_start'] = np.random.uniform(0, 70, home_mask.sum())  # Home attacks left side
        df.loc[~home_mask, 'x_start'] = np.random.uniform(35, 105, (~home_mask).sum())  # Away attacks right side
        
        print(f"✅ Created dummy data: {len(df)} events")
        return df
    
    def generate_heatmap_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate heatmap data using the EXACT original function"""
        print("🔄 Generating action heatmap data using df_match_action_heatmap...")
        heatmap_df = df_match_action_heatmap(df, pitch_length=105.0, pitch_width=68.0, length_edges=21, width_edges=14)  # Both teams
        
        if heatmap_df.empty:
            print("❌ No heatmap data generated")
            return pd.DataFrame()
        
        print(f"✅ Generated heatmap data: {len(heatmap_df)} zones")
        return heatmap_df
    
    def create_simplified_output(self, heatmap_df: pd.DataFrame, game_id: str, original_df: pd.DataFrame) -> Dict[str, Any]:
        """Create simplified output format: x, y, value (intensity) for each team"""
        print("🔄 Creating simplified output format...")
        
        # Get unique teams from the data
        teams = sorted(heatmap_df["team_id"].unique())
        
        # Initialize output structure
        output = {
            "game_id": game_id,
            "pitch_info": {
                "length_meters": self.pitch_length,
                "width_meters": self.pitch_width,
                "grid_size": f"{self.x_zones-1} x {self.y_zones-1}",
                "teams": [str(team) for team in teams]
            },
            "data": {}
        }
        
        # Process each team separately
        for team in teams:
            team_data = heatmap_df[heatmap_df["team_id"] == team].copy()
            
            # Create a complete grid for this team
            team_zones = []
            
            # Create 20x13 grid (x: 0-19, y: 0-12)
            for y in range(13):  # 0 to 12
                for x in range(20):  # 0 to 19
                    # Find matching zone data
                    zone_data = None
                    for _, row in team_data.iterrows():
                        # Calculate which grid cell this zone belongs to
                        x_cell = int(row["x_min"] // 5.25)  # 105/20 = 5.25
                        y_cell = int(row["y_min"] // 5.23)  # 68/13 ≈ 5.23
                        
                        if x_cell == x and y_cell == y:
                            zone_data = row
                            break
                    
                    # Create zone entry in simple format: x, y, value
                    zone_entry = {
                        "x": x,
                        "y": y,
                        "value": float(zone_data["intensity"]) if zone_data is not None else 0.0,
                        "alpha": float(zone_data["alpha"]) if zone_data is not None else 0.0,
                        "count": int(zone_data["count"]) if zone_data is not None else 0
                    }
                    team_zones.append(zone_entry)
            
            output["data"][str(team)] = team_zones
        
        print(f"✅ Created simplified output with {len(output['data'])} teams")
        return output
    
    def export_to_json(self, data: Dict[str, Any], game_id: str) -> str:
        """Export data to JSON file"""
        filename = f"action_heatmap_simple_{game_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported to: {filename}")
        return filename
    
    def print_summary(self, data: Dict[str, Any]):
        """Print a summary of the generated data"""
        print("\n" + "="*60)
        print("�� SIMPLIFIED ACTION HEATMAP DATA SUMMARY")
        print("="*60)
        
        teams = data["pitch_info"]["teams"]
        
        print(f"🎯 Game ID: {data['game_id']}")
        print(f"📏 Grid Size: {data['pitch_info']['grid_size']}")
        print(f"👥 Teams: {', '.join(teams)}")
        
        # Team statistics
        for team in teams:
            team_zones = data["data"][team]
            total_events = sum(z["count"] for z in team_zones)
            avg_value = np.mean([z["value"] for z in team_zones])
            avg_alpha = np.mean([z["alpha"] for z in team_zones])
            active_zones = sum(1 for z in team_zones if z["value"] > 0)
            
            print(f"   Team {team}: {len(team_zones)} zones, {total_events} events, avg value: {avg_value:.2f}, avg alpha: {avg_alpha:.3f}, active zones: {active_zones}")
        
        # Sample zones
        print(f"\n🎯 SAMPLE ZONES (x, y, value, alpha, count):")
        for team in teams[:1]:  # Show sample from first team only
            sample_zones = data["data"][team][:5]
            for zone in sample_zones:
                print(f"   Team {team}: x={zone['x']}, y={zone['y']}, value={zone['value']:.2f}, alpha={zone['alpha']:.3f}, count={zone['count']}")
        
        print(f"\n📋 FRONTEND USAGE:")
        print(f"   - Load the JSON file")
        print(f"   - Access team data: data.data[team_id]")
        print(f"   - Each zone: {x: 0, y: 0, value: 1.85, alpha: 0.095, count: 3}")
        print(f"   - Use x,y as grid coordinates (0-19, 0-12)")
        print(f"   - Use value for intensity, alpha for transparency")
        print(f"   - Fixed purple color with alpha transparency")

    def show_heatmap_figure(self, heatmap_df: pd.DataFrame, game_id: str):
        """
        Show the action heatmap figure (can be commented/uncommented)
        This function creates and displays the action heatmap visualization
        """
        try:
            # Import plotting functions
            from futiplot.soccer.pitch import plot_pitch
            from futiplot.utils import futicolor
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
            
            print("🖼️ Generating action heatmap figure...")
            
            # Create pitch
            fig, ax, pitch = plot_pitch(orientation="wide", figsize=(12, 8), linewidth=2)
            
            # Plot each zone
            for _, zone in heatmap_df.iterrows():
                x_min, x_max = zone["x_min"], zone["x_max"]
                y_min, y_max = zone["y_min"], zone["y_max"]
                alpha = zone["alpha"]
                count = zone["count"]
                team_id = zone["team_id"]
                
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
            ax.set_title(f"Action Heatmap - Game: {game_id}", fontsize=16, fontweight='bold')
            ax.set_xlabel("Length (meters)", fontsize=12)
            ax.set_ylabel("Width (meters)", fontsize=12)
            
            # Set axis limits
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            
            # Invert y-axis for proper football view
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.show()
            
            print("✅ Action heatmap displayed successfully!")
            
        except ImportError as e:
            print(f"❌ Could not import plotting libraries: {e}")
            print("💡 To enable plotting, install: matplotlib")
        except Exception as e:
            print(f"❌ Error generating plot: {e}")

def main():
    """Main function to run the simplified action heatmap generator"""
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("❌ Usage: python action_heatmap_generator_simple.py <game_id>")
        print("Example: python action_heatmap_generator_simple.py 4491")
        sys.exit(1)
    
    game_id = sys.argv[1]
    
    print("🚀 SIMPLIFIED ACTION HEATMAP GENERATOR")
    print("=" * 50)
    print(f"🎯 Target Game: {game_id}")
    
    # Initialize generator
    generator = SimpleActionHeatmapGenerator()
    
    try:
        # Load match data
        df = generator.load_match_data(game_id)
        
        # Generate heatmap data using EXACT original function
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
        
        # =====================================================================
        # UNCOMMENT THE LINE BELOW TO SHOW THE ACTION HEATMAP FIGURE
        # =====================================================================
        generator.show_heatmap_figure(heatmap_df, game_id)
        # =====================================================================
        
        print(f"\n🎉 SUCCESS! Simplified action heatmap data generated for game {game_id}")
        print(f"📁 Output file: {filename}")
        print(f"📊 Ready for frontend integration!")
        print(f"💡 To show the plot, uncomment the show_heatmap_figure line in the script")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
