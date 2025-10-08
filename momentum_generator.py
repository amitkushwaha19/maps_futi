#!/usr/bin/env python3
"""
Momentum Generator for Frontend Team
===================================

This script generates momentum chart data for a specific game_id and exports it as JSON
for easy frontend consumption. The momentum chart shows possession momentum over time
with bars, overlays, and time ticks.

Usage:
    python momentum_generator.py <game_id>

Output:
    - JSON file with momentum data
    - Optional figure display (uncomment show_momentum_figure())
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPLPath  # FIXED: Import with alias to avoid conflict
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from futiplot.match_card.match_card_charts import df_match_momentum, compute_timing_meta
    from futiplot.utils import load_sample_data
    from futiplot.utils import futicolor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


def _rounded_end_bar(ax, x, w, y0, y1, color, z=2, alpha=1.0):
    """Create a rounded-end bar (matching original implementation)"""
    if abs(y1 - y0) < 1e-6 or w <= 0:
        return
    
    r = min(w, abs(y1 - y0)) * 0.15
    k = 0.5522847498307936  # 4/3 * (sqrt(2) - 1)
    
    verts = []
    codes = []
    
    if y1 > y0:  # upward bar
        y_low, y_high = y0, y1
    else:  # downward bar
        y_low, y_high = y1, y0
    
    # Create rounded rectangle path - FIXED: All MPLPath references
    verts.append((x, y_low + r));           codes.append(MPLPath.MOVETO)
    verts += [(x, y_low + r - k*r), (x + r - k*r, y_low), (x + r, y_low)]
    codes += [MPLPath.CURVE4, MPLPath.CURVE4, MPLPath.CURVE4]
    verts.append((x + w - r, y_low));       codes.append(MPLPath.LINETO)
    verts += [(x + w - r + k*r, y_low), (x + w, y_low + r - k*r), (x + w, y_low + r)]
    codes += [MPLPath.CURVE4, MPLPath.CURVE4, MPLPath.CURVE4]
    verts.append((x + w, y_high));          codes.append(MPLPath.LINETO)
    verts.append((x, y_high));              codes.append(MPLPath.LINETO)
    verts.append((x, y_low + r));           codes.append(MPLPath.CLOSEPOLY)

    patch = PathPatch(MPLPath(verts, codes), facecolor=color, edgecolor="none", zorder=z, alpha=float(alpha))
    ax.add_patch(patch)


class MomentumGenerator:
    """Generate momentum chart data for a specific game_id"""
    
    def __init__(self, game_id):
        self.game_id = game_id
        self.momentum_data = None
        self.original_df = None
        self.meta = None
        
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
            
            # Check for possession actions
            possession_actions = {"pass", "dribble", "carry", "reception", "shot"}
            possession_events = df_game[df_game["type_name"].isin(possession_actions)]
            print(f"⚽ Found {len(possession_events)} possession events")
            
            if possession_events.empty:
                print("⚠️ No possession events found in this game")
                return False
                
            self.original_df = df_game
            
            # Compute timing metadata with plot gaps
            self.meta, _ = compute_timing_meta(df_game, bin_seconds=120, plot_gap_seconds=0.7 * 120)
            print(f"⏱️ Computed timing metadata")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def generate_momentum_data(self):
        """Generate momentum data using the original function"""
        print("📊 Generating momentum data...")
        
        try:
            # Use the original function with same parameters as test_match_charts.py
            self.momentum_data = df_match_momentum(
                self.original_df,
                bin_seconds=120,
                percentiles_path=None,
                smoothing=0.5,
                width_scale=0.85,
                meta=self.meta,
            )
            
            if self.momentum_data.empty:
                print("❌ No momentum data generated")
                return False
                
            # Count different types
            types = self.momentum_data['type'].value_counts()
            print(f"✅ Generated momentum data:")
            for t, count in types.items():
                print(f"   {t}: {count} items")
                
            return True
            
        except Exception as e:
            print(f"❌ Error generating momentum data: {e}")
            return False
    
    def create_frontend_output(self):
        """Create simplified output for frontend team"""
        print("📊 Creating frontend-friendly output...")
        
        if self.momentum_data is None or self.momentum_data.empty:
            print("❌ No momentum data available")
            return None
            
        # Create simplified structure
        output = {
            "game_id": str(self.game_id),
            "chart_info": {
                "x_axis": "Time (0-100%)",
                "y_axis": "Momentum (0-100%)",
                "center_line": 50.0,
                "bin_seconds": 120
            },
            "data": {
                "base_bars": [],
                "overlay_bars": [],
                "ticks": []
            }
        }
        
        # Process each data point by type
        for _, row in self.momentum_data.iterrows():
            data_point = {
                "type": str(row["type"]),
                "x0": float(row["x0"]) if pd.notna(row["x0"]) else None,
                "x1": float(row["x1"]) if pd.notna(row["x1"]) else None,
                "y0": float(row["y0"]) if pd.notna(row["y0"]) else None,
                "y1": float(row["y1"]) if pd.notna(row["y1"]) else None,
                "team_id": int(row["team_id"]) if pd.notna(row["team_id"]) else None,
                "home": str(row["home"]) if pd.notna(row["home"]) else None,
                "period_id": int(row["period_id"]) if pd.notna(row["period_id"]) else None,
                "time_bin": int(row["time_bin"]) if pd.notna(row["time_bin"]) else None,
                "bar_color": str(row["bar_color"]) if pd.notna(row["bar_color"]) else None,
                "bar_fill": str(row["bar_fill"]) if pd.notna(row["bar_fill"]) else None,
                "pctl": float(row["pctl"]) if pd.notna(row["pctl"]) else None,
                "x": float(row["x"]) if pd.notna(row["x"]) else None,
                "label": str(row["label"]) if pd.notna(row["label"]) else None
            }
            
            # Add to appropriate category
            if row["type"] == "bar":
                output["data"]["base_bars"].append(data_point)
            elif row["type"] == "bar_top":
                output["data"]["overlay_bars"].append(data_point)
            elif row["type"] == "tick":
                output["data"]["ticks"].append(data_point)
        
        print(f"✅ Created frontend data:")
        print(f"   Base bars: {len(output['data']['base_bars'])}")
        print(f"   Overlay bars: {len(output['data']['overlay_bars'])}")
        print(f"   Ticks: {len(output['data']['ticks'])}")
        
        return output
    
    def export_to_json(self, output_data, filename=None):
        """Export data to JSON file"""
        if filename is None:
            filename = f"momentum_data_{self.game_id}.json"
            
        filepath = Path(__file__).parent / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"�� Exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error exporting JSON: {e}")
            return None
    
    def print_summary(self, output_data):
        """Print summary of generated data"""
        print("\n" + "="*60)
        print("📊 MOMENTUM DATA SUMMARY")
        print("="*60)
        
        if not output_data or not output_data.get("data"):
            print("❌ No data to summarize")
            return
            
        data = output_data["data"]
        base_bars = data["base_bars"]
        overlay_bars = data["overlay_bars"]
        
        print(f"🎯 Game ID: {output_data['game_id']}")
        print(f"📊 Base Bars: {len(base_bars)}")
        print(f"�� Overlay Bars: {len(overlay_bars)}")
        print(f"⏰ Time Ticks: {len(data['ticks'])}")
        
        # Team breakdown
        if base_bars:
            teams = {}
            for bar in base_bars:
                if bar["team_id"] is not None:
                    team_id = bar["team_id"]
                    if team_id not in teams:
                        teams[team_id] = {"home": bar["home"], "bars": 0}
                    teams[team_id]["bars"] += 1
            
            print(f"\n�� TEAM BREAKDOWN:")
            for team_id, stats in teams.items():
                print(f"   Team {team_id} ({stats['home']}): {stats['bars']} base bars")
        
        # Period breakdown
        if base_bars:
            periods = {}
            for bar in base_bars:
                if bar["period_id"] is not None:
                    period = bar["period_id"]
                    if period not in periods:
                        periods[period] = 0
                    periods[period] += 1
            
            print(f"\n⏱️ PERIOD BREAKDOWN:")
            for period, count in sorted(periods.items()):
                print(f"   Period {period}: {count} bars")
    
    def show_momentum_figure(self):
        """Display the momentum figure (matching original implementation)"""
        print("🎨 Displaying momentum figure (original style)...")
        
        if self.momentum_data is None or self.momentum_data.empty:
            print("❌ No momentum data to display")
            return
            
        try:
            # Create figure matching original
            fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=futicolor.dark)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            
            # Center line behind bars
            ax.axhline(50, lw=1, color="#BBBBBB", zorder=0.1)
            
            # Time ticks behind
            ticks = self.momentum_data[self.momentum_data['type'] == 'tick']
            if not ticks.empty:
                for _, r in ticks.iterrows():
                    ax.axvline(r["x"], lw=0.6, color="#DDDDDD", zorder=0)
                    if r["label"] == "HT":
                        ax.text(r["x"], 98, "HT", ha="center", va="top", 
                               fontsize=8, color="#777777", zorder=4)
            
            # Base bars (rounded far end)
            base_bars = self.momentum_data[self.momentum_data['type'] == 'bar']
            if not base_bars.empty:
                for _, r in base_bars.iterrows():
                    x = float(r["x0"])
                    w = float(r["x1"] - r["x0"])
                    y0, y1 = float(r["y0"]), float(r["y1"])
                    if abs(y1 - y0) < 1e-6 or w <= 0:
                        continue
                    _rounded_end_bar(ax, x=x, w=w, y0=y0, y1=y1, color=r["bar_color"], z=2, alpha=0.5)
            
            # Overlay bars (rounded far end)
            overlay_bars = self.momentum_data[self.momentum_data['type'] == 'bar_top']
            if not overlay_bars.empty:
                for _, r in overlay_bars.iterrows():
                    x = float(r["x0"])
                    w = float(r["x1"] - r["x0"])
                    y0, y1 = float(r["y0"]), float(r["y1"])
                    if abs(y1 - y0) < 1e-6 or w <= 0:
                        continue
                    _rounded_end_bar(ax, x=x, w=w, y0=y0, y1=y1, color=r["bar_fill"], z=3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ Error displaying figure: {e}")


def main():
    """Main function to run the momentum generator"""
    if len(sys.argv) != 2:
        print("Usage: python momentum_generator.py <game_id>")
        print("Example: python momentum_generator.py 4491")
        sys.exit(1)
    
    game_id = sys.argv[1]
    
    print("�� MOMENTUM GENERATOR")
    print("="*50)
    
    # Create generator
    generator = MomentumGenerator(game_id)
    
    # Load data
    if not generator.load_match_data():
        sys.exit(1)
    
    # Generate momentum data
    if not generator.generate_momentum_data():
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
    generator.show_momentum_figure()
    
    print(f"\n✅ Momentum generation complete!")
    print(f"📁 JSON file: {json_file}")
    print(f"�� To show figure, uncomment: generator.show_momentum_figure()")


if __name__ == "__main__":
    main()