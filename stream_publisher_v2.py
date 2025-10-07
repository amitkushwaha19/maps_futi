#!/usr/bin/env python3
"""
Stream Publisher v2 - Publishes VAEP events to per-match Redis Streams with Deduplication
=========================================================================================

Aligns with Redis Stream Schema Document:
- Stream naming: stream:match:{match_id}:events
- Each VAEP action is a separate stream entry
- All CSV columns preserved as stream fields
- Consumes from consumer's output and publishes to per-match streams
- DEDUPLICATION: Prevents duplicate VAEP events in streams
"""

import os
import sys
import json
import logging
import hashlib
from typing import Dict, Any, Set

import pandas as pd
import redis
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [STREAM_PUBLISHER_V2] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("stream_publisher_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# Deduplication settings
PUBLISHED_EVENTS_TTL = 86400 * 7  # 7 days TTL for published events tracking


# -------------------- Deduplication Functions --------------------

def get_stream_name(match_id: int) -> str:
    """Generate stream name following the schema pattern"""
    return f"stream:match:{match_id}:events"


def get_published_events_key(match_id: int) -> str:
    """Get Redis key for tracking published events for a match"""
    return f"published_events:{match_id}"


def get_event_hashes_key(match_id: int) -> str:
    """Get Redis key for tracking event content hashes"""
    return f"published_hashes:{match_id}"


def generate_vaep_event_key(row: pd.Series) -> str:
    """
    Generate unique key for a VAEP event
    Uses action_id and time_seconds as primary identifiers
    """
    match_id = row.get('match_id', 0)
    action_id = row.get('action_id', 0)
    time_seconds = row.get('time_seconds', 0)
    player_id = row.get('player_id', 'unknown')
    type_name = row.get('type_name', 'unknown')
    
    return f"{match_id}:{action_id}:{time_seconds}:{player_id}:{type_name}"


def generate_vaep_event_hash(row: pd.Series) -> str:
    """
    Generate content hash for a VAEP event to detect changes
    """
    # Critical fields that define the event content
    critical_fields = [
        'action_id', 'time_seconds', 'player_id', 'team_id', 'type_name',
        'result_name', 'x_start', 'y_start', 'x_end', 'y_end',
        'vaep', 'offensive_vaep', 'defensive_vaep', 'prob_score', 'prob_concede'
    ]
    
    event_content = ""
    for field in critical_fields:
        if field in row:
            value = row[field]
            if pd.isna(value):
                event_content += f"{field}:None"
            else:
                event_content += f"{field}:{str(value)}"
    
    return hashlib.md5(event_content.encode()).hexdigest()


def filter_new_vaep_events(match_id: int, vaep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out VAEP events that have already been published
    Returns only new or changed events
    """
    if vaep_df is None or len(vaep_df) == 0:
        return vaep_df
    
    published_key = get_published_events_key(match_id)
    hashes_key = get_event_hashes_key(match_id)
    
    # Get already published event keys
    published_events = r.smembers(published_key)
    published_events = {key.decode() for key in published_events}
    
    # Get stored event hashes
    stored_hashes = r.hgetall(hashes_key)
    stored_hashes = {key.decode(): value.decode() for key, value in stored_hashes.items()}
    
    new_events = []
    updated_events = []
    
    for idx, row in vaep_df.iterrows():
        event_key = generate_vaep_event_key(row)
        event_hash = generate_vaep_event_hash(row)
        
        if event_key not in published_events:
            # Completely new event
            new_events.append(idx)
        elif stored_hashes.get(event_key) != event_hash:
            # Event exists but content changed (e.g., VAEP recalculated)
            updated_events.append(idx)
            logger.info(f"VAEP event {event_key} content changed, will republish")
    
    if not new_events and not updated_events:
        logger.info(f"No new or updated VAEP events for match {match_id}")
        return pd.DataFrame()
    
    filtered_indices = new_events + updated_events
    filtered_df = vaep_df.iloc[filtered_indices].copy()
    
    logger.info(f"Match {match_id}: {len(new_events)} new VAEP events, {len(updated_events)} updated events")
    
    return filtered_df


def mark_vaep_events_as_published(match_id: int, vaep_df: pd.DataFrame):
    """
    Mark VAEP events as published in Redis
    """
    if vaep_df is None or len(vaep_df) == 0:
        return
    
    published_key = get_published_events_key(match_id)
    hashes_key = get_event_hashes_key(match_id)
    
    # Use pipeline for atomic operations
    pipe = r.pipeline()
    
    for _, row in vaep_df.iterrows():
        event_key = generate_vaep_event_key(row)
        event_hash = generate_vaep_event_hash(row)
        
        # Add to published events set
        pipe.sadd(published_key, event_key)
        # Store event hash
        pipe.hset(hashes_key, event_key, event_hash)
    
    # Set TTL for cleanup
    pipe.expire(published_key, PUBLISHED_EVENTS_TTL)
    pipe.expire(hashes_key, PUBLISHED_EVENTS_TTL)
    
    pipe.execute()


# -------------------- Stream Publishing Functions --------------------

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
    Publish VAEP events for a match to its dedicated Redis Stream (with deduplication)
    
    Args:
        vaep_df: DataFrame with VAEP results
        match_id: Match identifier
    """
    stream_name = get_stream_name(match_id)
    
    # Filter out already published events
    new_vaep_df = filter_new_vaep_events(match_id, vaep_df)
    
    if len(new_vaep_df) == 0:
        logger.info(f"No new VAEP events to publish for match {match_id}")
        return
    
    logger.info(f"Publishing {len(new_vaep_df)} new VAEP events to {stream_name}")
    
    published_count = 0
    
    for idx, row in new_vaep_df.iterrows():
        try:
            # Prepare entry with all fields
            entry = prepare_stream_entry(row)
            
            # Add to stream (* = auto-generate ID)
            stream_id = r.xadd(stream_name, entry)
            published_count += 1
            
            # Log every 100 events
            if published_count % 100 == 0:
                logger.info(f"  Published {published_count}/{len(new_vaep_df)} events...")
                
        except Exception as e:
            logger.error(f"Failed to publish event {idx}: {e}")
            continue
    
    # Mark events as published
    mark_vaep_events_as_published(match_id, new_vaep_df)
    
    logger.info(f"✓ Published {published_count} new events to {stream_name}")
    
    # Update stream metadata
    metadata_key = f"stream:match:{match_id}:metadata"
    total_events = r.xlen(stream_name)
    r.hset(metadata_key, mapping={
        "match_id": str(match_id),
        "total_events": str(total_events),
        "stream_name": stream_name,
        "last_updated": pd.Timestamp.now().isoformat(),
        "new_events_added": str(published_count)
    })


def listen_vaep_output_channel():
    """
    Listen to vaep_output_channel and publish to per-match streams
    """
    pubsub = r.pubsub()
    pubsub.subscribe("vaep_output_channel")
    
    logger.info("=" * 60)
    logger.info("Stream Publisher v2 Started (with Deduplication)")
    logger.info("Listening on: vaep_output_channel")
    logger.info("Publishing to: stream:match:{match_id}:events")
    logger.info(f"Deduplication TTL: {PUBLISHED_EVENTS_TTL}s")
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
            
            # Publish to stream (with deduplication)
            publish_vaep_to_stream(vaep_df, match_id)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)


def load_csv_to_streams(csv_path: str):
    """
    Load existing CSV file and publish to Redis Streams (with deduplication)
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
    
    # 5. Get deduplication stats
    published_key = get_published_events_key(match_id)
    published_count = r.scard(published_key)
    logger.info(f"\n5. Deduplication stats:")
    logger.info(f"  Tracked published events: {published_count}")
    
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


def cleanup_old_tracking_data():
    """Clean up old event tracking data to prevent memory bloat"""
    logger.info("Cleaning up old event tracking data...")
    
    # Find all published_events and published_hashes keys
    cursor = 0
    cleaned_count = 0
    
    patterns = ["published_events:*", "published_hashes:*"]
    
    for pattern in patterns:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor, match=pattern, count=100)
            
            for key in keys:
                # Check if key has TTL, if not set it
                ttl = r.ttl(key)
                if ttl == -1:  # No TTL set
                    r.expire(key, PUBLISHED_EVENTS_TTL)
                    cleaned_count += 1
            
            if cursor == 0:
                break
    
    if cleaned_count > 0:
        logger.info(f"Set TTL for {cleaned_count} tracking keys")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="VAEP Stream Publisher v2 (with Deduplication)")
    parser.add_argument(
        "--mode",
        choices=["listen", "backfill", "query", "trim", "cleanup"],
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
        cleanup_old_tracking_data()
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
        
    elif args.mode == "cleanup":
        # Cleanup tracking data
        cleanup_old_tracking_data()


if __name__ == "__main__":
    main()