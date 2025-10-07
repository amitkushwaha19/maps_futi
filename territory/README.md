# Territory Plot Generator

## 📊 Overview
This folder contains the territory plot generator script and sample output for frontend integration.

## 📁 Files

### `territory_plot_generator.py`
- **Purpose**: Generates territory plot data for a specific match
- **Usage**: `python territory_plot_generator.py <game_id>`
- **Output**: JSON file with territory data in simplified format

### `territory_data_4491.json`
- **Purpose**: Sample output for game_id 4491
- **Format**: `{x, y, home_value, away_value, fill_color}`
- **Grid**: 6×5 zones (30 total zones)
- **Use**: Frontend integration example

## 🎯 Data Format
```json
{
  "zone_id": "(0,0)",
  "x": 0,                    // Grid X coordinate (0-5)
  "y": 0,                    // Grid Y coordinate (0-4)
  "home_value": 33,          // Home team percentage
  "away_value": 67,          // Away team percentage
  "fill_color": "#EA1F96"    // Color for visualization
}
```

## 🚀 Usage
```bash
cd generators/territory
python territory_plot_generator.py 4491
```

## 🎨 Frontend Integration
- Load JSON file
- Access zones: `data.zones[index]`
- Use `x, y` for grid positioning
- Use `home_value, away_value` for percentages
- Use `fill_color` for visualization
