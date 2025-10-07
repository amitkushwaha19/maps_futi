# Football Analytics Pipeline

A comprehensive real-time football analytics pipeline that processes StatsBomb event data through SPADL → VAEP transformation and generates interactive visualizations.

## 🏗️ Architecture

```
StatsBomb API → Producer → Redis Streams → Consumer → VAEP Processing → Redis Streams → Visualization Generators
```

## 📊 Features

### Data Pipeline
- **Producer**: Fetches live match events from StatsBomb API
- **Consumer**: Processes events through SPADL → VAEP pipeline
- **Stream Publisher**: Publishes processed data to per-match Redis streams
- **Deduplication**: Ensures data quality by removing duplicate events

### Visualization Generators
- **Momentum Charts**: Possession momentum over time with bars and overlays
- **Action Heatmaps**: Player/team action intensity across pitch zones
- **Shot Maps**: Shot locations with xG values and goal indicators
- **Territory Plots**: Possession dominance by pitch zones
- **xG Step Charts**: Cumulative expected goals progression

### Data Quality Tools
- **CSV Deduplicator**: Removes duplicate events from historical data
- **Stream Updater**: Updates Redis streams with clean data
- **Redis Generators**: All include built-in deduplication logic

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis Server
- StatsBomb API credentials

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd football-analytics-pipeline
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your StatsBomb credentials
```

5. Start Redis server:
```bash
redis-server
```

### Running the Pipeline

1. **Start the Producer** (fetches live data):
```bash
python producer.py
```

2. **Start the Consumer** (processes data):
```bash
python consumer.py
```

3. **Start the Stream Publisher** (publishes to Redis):
```bash
python stream_publisher.py
```

4. **Generate Visualizations**:
```bash
# Generate all visualizations for all matches
python momentum/momentum_generator.py
python action_heatmap/action_heatmap_generator_redis.py
python shotmap/shotmap_generator_redis.py
python territory/territory_generator_redis.py
python xg_step/xg_step_generator_redis.py
```

## 🧹 Data Quality Management

### Clean Historical Data
```bash
# Remove duplicates from CSV
python csv_deduplicator.py --input vaep_output_data.csv --output vaep_output_data_clean.csv

# Update Redis streams with clean data
python stream_updater.py --csv vaep_output_data_clean.csv --update-all
```

See [DEDUPLICATION_GUIDE.md](DEDUPLICATION_GUIDE.md) for detailed instructions.

## 📁 Project Structure

```
├── producer.py                 # Fetches live events from StatsBomb API
├── consumer.py                 # SPADL → VAEP processing pipeline
├── stream_publisher.py         # Publishes to per-match Redis streams
├── csv_deduplicator.py        # Removes duplicate events from CSV
├── stream_updater.py          # Updates Redis streams with clean data
├── functions/                 # Core processing functions
│   ├── sb_api_functions.py    # StatsBomb API integration
│   ├── spadl_processing_functions.py  # SPADL conversion
│   ├── vaep_functions.py      # VAEP calculation
│   └── db_connect.py          # Database utilities
├── momentum/                  # Momentum chart generation
├── action_heatmap/           # Action heatmap generation
├── shotmap/                  # Shot map generation
├── territory/                # Territory plot generation
├── xg_step/                  # xG step chart generation
├── futiplot/                 # Visualization library
├── models/                   # ML models for VAEP
└── requirements.txt          # Python dependencies
```

## 🎯 Generated Outputs

Each visualization generator creates JSON files optimized for frontend consumption:

- `momentum_data_{match_id}.json` - Momentum chart data
- `action_heatmap_data_{match_id}.json` - Heatmap zone data
- `shotmap_data_{match_id}.json` - Shot location and xG data
- `territory_data_{match_id}.json` - Territory dominance data
- `xg_step_data_{match_id}.json` - Cumulative xG progression

## 🔧 Configuration

### Environment Variables (.env)
```bash
SB_CLIENT_ID=your_statsbomb_client_id
SB_CLIENT_SECRET=your_statsbomb_client_secret
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Redis Streams Schema
- **Live Events**: `live_events_stream`
- **Per-Match Events**: `stream:match:{match_id}:events`
- **Metadata**: `stream:match:{match_id}:metadata`

## 🛠️ Development

### Adding New Visualizations
1. Create new generator in appropriate directory
2. Follow the Redis generator pattern (see existing generators)
3. Include deduplication logic in `load_match_data()`
4. Export frontend-friendly JSON format

### Testing
```bash
# Test individual components
python -m pytest tests/

# Test specific generator
python momentum/momentum_generator.py
```

## 📈 Performance

- **Producer**: Processes ~1000 events/minute
- **Consumer**: VAEP calculation ~500 events/minute
- **Generators**: ~100K events/minute per visualization
- **Deduplication**: ~100K events/minute

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **StatsBomb** for providing football event data
- **VAEP** methodology for action valuation
- **SPADL** for standardized event representation
- **Futiplot** for visualization utilities

## 📞 Support

For questions or issues:
1. Check the [DEDUPLICATION_GUIDE.md](DEDUPLICATION_GUIDE.md) for data quality issues
2. Review existing issues on GitHub
3. Create new issue with detailed description

---

**Built with ⚽ for football analytics enthusiasts**