#!/usr/bin/env python3
"""
Stream Publisher - Publishes VAEP events to per-match Redis Streams
====================================================================

Aligns with Redis Stream Schema Document:
- Stream naming: stream:match:{match_id}:events
- Each VAEP action is a separate stream entry
- All CSV columns preserved as stream fields
- Consumes from consumer's output and publishes to per-match streams
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import pandas as pd
import redis
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [STREAM_PUBLISHER] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stream_publisher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)


def get_stream_name(match_id: int) -> str:
    """Generate stream name following the schema pattern"""
    return f"stream:match:{match_id}:events"


def prepare_stream_entry(row: pd.Series) -> Dict[str, str]:
    """
    Convert a VAEP dataframe row to Redis Stream entry format
    All values converted to strings, None/NaN becomes empty string
    """
    entry = {}
    
    for key, value in row.items():
        # Convert to string, handle None/NaN
        if pd.isna(value):
            entry[str(key)] = ""
        elif isinstance(value, (int, float)):
            entry[str(key)] = str(value)
        elif isinstance(value, bool):
            entry[str(key)] = "TRUE" if value else "FALSE"
        else:
            entry[str(key)] = str(value)
    
    return entry


def publish_vaep_to_stream(vaep_df: pd.DataFrame, match_id: int):
    """
    Publish all VAEP events for a match to its dedicated Redis Stream
    
    Args:
        vaep_df: DataFrame with VAEP results
        match_id: Match identifier
    """
    stream_name = get_stream_name(match_id)
    
    logger.info(f"Publishing {len(vaep_df)} events to {stream_name}")
    
    published_count = 0
    
    for idx, row in vaep_df.iterrows():
        try:
            # Prepare entry with all fields
            entry = prepare_stream_entry(row)
            
            # Add to stream (* = auto-generate ID)
            stream_id = r.xadd(stream_name, entry)
            published_count += 1
            
            # Log every 100 events
            if published_count % 100 == 0:
                logger.info(f"  Published {published_count}/{len(vaep_df)} events...")
                
        except Exception as e:
            logger.error(f"Failed to publish event {idx}: {e}")
            continue
    
    logger.info(f"✓ Published {published_count} events to {stream_name}")
    
    # Set stream metadata
    metadata_key = f"stream:match:{match_id}:metadata"
    r.hset(metadata_key, mapping={
        "match_id": str(match_id),
        "total_events": str(published_count),
        "stream_name": stream_name,
        "last_updated": pd.Timestamp.now().isoformat()
    })


def listen_vaep_output_channel():
    """
    Listen to vaep_output_channel and publish to per-match streams
    """
    pubsub = r.pubsub()
    pubsub.subscribe("vaep_output_channel")
    
    logger.info("=" * 60)
    logger.info("Stream Publisher Started")
    logger.info("Listening on: vaep_output_channel")
    logger.info("Publishing to: stream:match:{match_id}:events")
    logger.info("=" * 60)
    
    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        
        try:
            # Parse VAEP data
            data = msg["data"].decode("utf-8")
            vaep_df = pd.read_json(data, orient="records")
            
            if len(vaep_df) == 0:
                logger.warning("Received empty VAEP data")
                continue
            
            # Extract match_id
            if "match_id" not in vaep_df.columns:
                logger.error("No match_id in VAEP data")
                continue
            
            match_id = int(vaep_df["match_id"].iloc[0])
            
            # Publish to stream
            publish_vaep_to_stream(vaep_df, match_id)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)


def load_csv_to_streams(csv_path: str):
    """
    Load existing CSV file and publish to Redis Streams
    Useful for backfilling historical data
    
    Args:
        csv_path: Path to VAEP CSV file
    """
    logger.info(f"Loading CSV from {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} total events")
    
    # Group by match_id
    for match_id, group in df.groupby("match_id"):
        logger.info(f"Processing match {match_id} ({len(group)} events)")
        publish_vaep_to_stream(group, int(match_id))
    
    logger.info("✓ CSV backfill complete")


def query_stream_examples(match_id: int):
    """
    Example queries for reading from streams
    
    Args:
        match_id: Match identifier
    """
    stream_name = get_stream_name(match_id)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Example Queries for {stream_name}")
    logger.info(f"{'='*60}\n")
    
    # 1. Get stream length
    stream_len = r.xlen(stream_name)
    logger.info(f"1. Stream length: {stream_len} events")
    
    # 2. Read first 5 events
    logger.info("\n2. First 5 events:")
    first_events = r.xrange(stream_name, count=5)
    for event_id, fields in first_events:
        logger.info(f"  ID: {event_id.decode()}")
        logger.info(f"  Action: {fields.get(b'type_name', b'').decode()}")
        logger.info(f"  VAEP: {fields.get(b'vaep', b'').decode()}")
    
    # 3. Read last 5 events
    logger.info("\n3. Last 5 events (newest first):")
    last_events = r.xrevrange(stream_name, count=5)
    for event_id, fields in last_events:
        logger.info(f"  ID: {event_id.decode()}")
        logger.info(f"  Action: {fields.get(b'type_name', b'').decode()}")
    
    # 4. Get metadata
    metadata_key = f"stream:match:{match_id}:metadata"
    metadata = r.hgetall(metadata_key)
    if metadata:
        logger.info("\n4. Stream metadata:")
        for key, value in metadata.items():
            logger.info(f"  {key.decode()}: {value.decode()}")
    
    logger.info(f"\n{'='*60}\n")


def trim_old_streams(max_length: int = 10000):
    """
    Trim all match streams to a maximum length to save memory
    
    Args:
        max_length: Maximum number of events to keep per stream
    """
    logger.info(f"Trimming streams to max length: {max_length}")
    
    # Find all match streams
    pattern = "stream:match:*:events"
    cursor = 0
    trimmed_count = 0
    
    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=100)
        
        for key in keys:
            try:
                # Trim stream
                r.xtrim(key, maxlen=max_length, approximate=True)
                trimmed_count += 1
                
                if trimmed_count % 10 == 0:
                    logger.info(f"  Trimmed {trimmed_count} streams...")
                    
            except Exception as e:
                logger.error(f"Failed to trim {key.decode()}: {e}")
        
        if cursor == 0:
            break
    
    logger.info(f"✓ Trimmed {trimmed_count} streams")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VAEP Stream Publisher")
    parser.add_argument(
        "--mode",
        choices=["listen", "backfill", "query", "trim"],
        default="listen",
        help="Operation mode"
    )
    parser.add_argument(
        "--csv",
        help="CSV file path (for backfill mode)"
    )
    parser.add_argument(
        "--match-id",
        type=int,
        help="Match ID (for query mode)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10000,
        help="Max stream length (for trim mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "listen":
        # Real-time listening mode
        listen_vaep_output_channel()
        
    elif args.mode == "backfill":
        # Backfill from CSV
        if not args.csv:
            logger.error("--csv required for backfill mode")
            return
        load_csv_to_streams(args.csv)
        
    elif args.mode == "query":
        # Query examples
        if not args.match_id:
            logger.error("--match-id required for query mode")
            return
        query_stream_examples(args.match_id)
        
    elif args.mode == "trim":
        # Trim streams
        trim_old_streams(args.max_length)


if __name__ == "__main__":
    main()