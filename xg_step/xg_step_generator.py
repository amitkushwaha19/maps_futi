#!/usr/bin/env python3
"""
xG Step Generator for Frontend Team (Fixed to match original)
============================================================

This script generates xG step chart data for a specific game_id and exports it as JSON
for easy frontend consumption. The xG step chart shows cumulative expected goals over time
with step lines, goal markers, and time ticks.

Usage:
    python xg_step_generator_fixed.py <game_id>

Output:
    - JSON file with xG step data
    - Optional figure display (uncomment show_xg_step_figure())
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from futiplot.match_card.match_card_charts import df_match_xg_step, compute_timing_meta
    from futiplot.utils import load_sample_data
    from futiplot.utils import futicolor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class XGStepGenerator:
    """Generate xG step chart data for a specific game_id"""
    
    def __init__(self, game_id):
        self.game_id = game_id
        self.xg_step_data = None
        self.original_df = None
        self.meta = None
        self.per_bounds = None
        
    def load_match_data(self):
        """Load match data and compute timing metadata"""
        print(f"🔍 Loading data for game_id: {self.game_id}")
        
        try:
            # Load sample data
            df = load_sample_data('vaep')
            
            # Convert game_id to int for comparison
            if isinstance(self.game_id, str):
                game_id_int = int(self.game_id)
            else:
                game_id_int = self.game_id
                
            # Filter to specific game
            df_game = df[df['game_id'] == game_id_int].copy()
            
            if df_game.empty:
                print(f"❌ No data found for game_id: {self.game_id}")
                return False
                
            print(f"✅ Found {len(df_game)} events for game {self.game_id}")
            
            # Check for shots
            shots_raw = df_game[df_game.get('xg', 0) > 0]
            print(f"🏹 Found {len(shots_raw)} shots")
            
            if shots_raw.empty:
                print("⚠️ No shots found in this game")
                return False
                
            self.original_df = df_game
            
            # Compute timing metadata
            self.meta, self.per_bounds = compute_timing_meta(df_game, bin_seconds=120, plot_gap_seconds=None)
            print(f"⏱️ Computed timing metadata")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_xg_step_data(self):
        """Generate xG step data using the original function"""
        print("📈 Generating xG step data...")
        
        try:
            # Use the original function
            self.xg_step_data = df_match_xg_step(self.original_df, meta=self.meta, per_bounds=self.per_bounds)
            
            if self.xg_step_data.empty:
                print("❌ No xG step data generated")
                return False
                
            # Count different types
            types = self.xg_step_data['type'].value_counts()
            print(f"✅ Generated xG step data:")
            for t, count in types.items():
                print(f"   {t}: {count} points")
                
            return True
            
        except Exception as e:
            print(f"❌ Error generating xG step data: {e}")
            return False
    
    def create_frontend_output(self):
        """Create simplified output for frontend team"""
        print("📊 Creating frontend-friendly output...")
        
        if self.xg_step_data is None or self.xg_step_data.empty:
            print("❌ No xG step data available")
            return None
            
        # Create simplified structure
        output = {
            "game_id": str(self.game_id),
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
        
        # Process each data point by type
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
            filename = f"xg_step_data_{self.game_id}.json"
            
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
        
        print(f"🎯 Game ID: {output_data['game_id']}")
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
    
    def show_xg_step_figure(self):
        """Display the xG step figure (matching original implementation)"""
        print("🎨 Displaying xG step figure (original style)...")
        
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
                    if r["label"] == "HT":
                        ax.text(r["x"], 98, "HT", ha="center", va="top", 
                               fontsize=8, color="#888888", zorder=4)
            
            # Step lines by team (matching original)
            steps = self.xg_step_data[self.xg_step_data['type'] == 'shot']
            if not steps.empty:
                for (_, tid), g in steps.groupby(["game_id", "team_id"], sort=False):
                    g = g.sort_values("x", kind="stable")
                    ax.plot(g["x"], g["y"], lw=2.0, color=g["line_color"].iloc[0],
                           drawstyle="steps-post", zorder=2)
            
            # Goal markers + labels (matching original)
            goals = self.xg_step_data[self.xg_step_data['type'] == 'goal']
            if not goals.empty:
                ax.scatter(goals["x"], goals["y"], s=100, c=goals["point_color"],
                          edgecolors=futicolor.dark, linewidths=1, zorder=3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying figure: {e}")


def main():
    """Main function to run the xG step generator"""
    if len(sys.argv) != 2:
        print("Usage: python xg_step_generator_fixed.py <game_id>")
        print("Example: python xg_step_generator_fixed.py 4491")
        sys.exit(1)
    
    game_id = sys.argv[1]
    
    print("📈 xG STEP GENERATOR (FIXED)")
    print("="*50)
    
    # Create generator
    generator = XGStepGenerator(game_id)
    
    # Load data
    if not generator.load_match_data():
        sys.exit(1)
    
    # Generate xG step data
    if not generator.generate_xg_step_data():
        sys.exit(1)
    
    # Create frontend output
    output_data = generator.create_frontend_output()
    if output_data is None:
        sys.exit(1)
    
    # Export to JSON
    json_file = generator.export_to_json(output_data)
    if json_file is None:
        sys.exit(1)
    
    # Print summary
    generator.print_summary(output_data)
    
    # Show figure (uncomment to display)
    generator.show_xg_step_figure()
    
    print(f"\n✅ xG step generation complete!")
    print(f"📁 JSON file: {json_file}")
    print(f"🎨 Figure now matches original implementation!")


if __name__ == "__main__":
    main()
