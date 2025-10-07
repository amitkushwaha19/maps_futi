#!/usr/bin/env python3
"""
Event Producer v2 - Pushes raw StatsBomb events to Redis Streams with Deduplication
==================================================================================

Features:
- Fetches matches in UTC rolling window [yesterday .. tomorrow+3]
- Polls live events via GraphQL API
- Pushes raw events to Redis Stream: 'live_events_stream'
- Each message contains: match_id, match_info, raw_events (JSON)
- Listens to 'new_match_channel' for immediate triggers
- DEDUPLICATION: Tracks processed events to avoid duplicates
"""

import os
import sys
import time
import json
import logging
import threading
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Set

import pandas as pd
import redis
from dotenv import load_dotenv

# Import API functions
sys.path.append(os.path.abspath(os.path.join('.', 'functions')))
from functions.sb_api_functions import create_sblive_client, fetch_matches_by_date, fetch_live_match_event,fetch_access_token

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PRODUCER_V2] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("producer_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# API Client
SB_CLIENT_ID = os.getenv("SB_CLIENT_ID")
SB_CLIENT_SECRET = os.getenv("SB_CLIENT_SECRET")

ACCESS_TOKEN = fetch_access_token(SB_CLIENT_ID, SB_CLIENT_SECRET)
client = create_sblive_client(ACCESS_TOKEN)

# Track matches we've seen to avoid duplicate pushes
seen_matches = set()
STREAM_NAME = "live_events_stream"

# Deduplication settings
PROCESSED_EVENTS_TTL = 86400 * 7  # 7 days TTL for processed events tracking


def _get_league_name(competition_id: int) -> str:
    """Map competition ID to league name"""
    mapping = {
        44: "MLS", 37: "Premier League", 2: "Champions League",
        11: "La Liga", 9: "Serie A", 8: "Bundesliga", 4: "Ligue 1"
    }
    return mapping.get(int(competition_id) if competition_id is not None else competition_id,
                       f"Competition_{competition_id}")


def generate_event_key(match_id: int, event_row: pd.Series) -> str:
    """
    Generate unique key for an event
    Uses event_id if available, otherwise creates composite key
    """
    if 'id' in event_row and pd.notna(event_row['id']):
        return f"{match_id}:{event_row['id']}"
    
    # Fallback composite key
    minute = event_row.get('minute', 0)
    second = event_row.get('second', 0)
    player_id = event_row.get('player_id', 'unknown')
    type_name = event_row.get('type', {}).get('name', 'unknown') if isinstance(event_row.get('type'), dict) else str(event_row.get('type', 'unknown'))
    x_start = event_row.get('location', [0, 0])[0] if isinstance(event_row.get('location'), list) and len(event_row.get('location', [])) > 0 else 0
    y_start = event_row.get('location', [0, 0])[1] if isinstance(event_row.get('location'), list) and len(event_row.get('location', [])) > 1 else 0
    
    return f"{match_id}:{minute}:{second}:{player_id}:{type_name}:{x_start}:{y_start}"


def generate_event_hash(event_row: pd.Series) -> str:
    """
    Generate content hash for an event to detect changes
    """
    # Create a string representation of critical event data
    critical_fields = ['minute', 'second', 'type', 'player', 'team', 'location', 'pass', 'shot', 'duel']
    event_content = ""
    
    for field in critical_fields:
        if field in event_row:
            event_content += f"{field}:{str(event_row[field])}"
    
    return hashlib.md5(event_content.encode()).hexdigest()


def get_processed_events_key(match_id: int) -> str:
    """Get Redis key for tracking processed events for a match"""
    return f"processed_events:{match_id}"


def get_event_hashes_key(match_id: int) -> str:
    """Get Redis key for tracking event content hashes"""
    return f"event_hashes:{match_id}"


def filter_new_events(match_id: int, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out events that have already been processed
    Returns only new or changed events
    """
    if events_df is None or len(events_df) == 0:
        return events_df
    
    processed_key = get_processed_events_key(match_id)
    hashes_key = get_event_hashes_key(match_id)
    
    # Get already processed event keys
    processed_events = r.smembers(processed_key)
    processed_events = {key.decode() for key in processed_events}
    
    # Get stored event hashes
    stored_hashes = r.hgetall(hashes_key)
    stored_hashes = {key.decode(): value.decode() for key, value in stored_hashes.items()}
    
    new_events = []
    updated_events = []
    
    for idx, event_row in events_df.iterrows():
        event_key = generate_event_key(match_id, event_row)
        event_hash = generate_event_hash(event_row)
        
        if event_key not in processed_events:
            # Completely new event
            new_events.append(idx)
        elif stored_hashes.get(event_key) != event_hash:
            # Event exists but content changed
            updated_events.append(idx)
            logger.info(f"Event {event_key} content changed, will reprocess")
    
    if not new_events and not updated_events:
        logger.info(f"No new or updated events for match {match_id}")
        return pd.DataFrame()
    
    filtered_indices = new_events + updated_events
    filtered_df = events_df.iloc[filtered_indices].copy()
    
    logger.info(f"Match {match_id}: {len(new_events)} new events, {len(updated_events)} updated events")
    
    return filtered_df


def mark_events_as_processed(match_id: int, events_df: pd.DataFrame):
    """
    Mark events as processed in Redis
    """
    if events_df is None or len(events_df) == 0:
        return
    
    processed_key = get_processed_events_key(match_id)
    hashes_key = get_event_hashes_key(match_id)
    
    # Use pipeline for atomic operations
    pipe = r.pipeline()
    
    for _, event_row in events_df.iterrows():
        event_key = generate_event_key(match_id, event_row)
        event_hash = generate_event_hash(event_row)
        
        # Add to processed events set
        pipe.sadd(processed_key, event_key)
        # Store event hash
        pipe.hset(hashes_key, event_key, event_hash)
    
    # Set TTL for cleanup
    pipe.expire(processed_key, PROCESSED_EVENTS_TTL)
    pipe.expire(hashes_key, PROCESSED_EVENTS_TTL)
    
    pipe.execute()


def push_events_to_stream(match_id: int, match_info: Dict, events_df: pd.DataFrame) -> bool:
    """
    Push raw events to Redis Stream (only new/changed events)
    
    Returns:
        bool: True if events were pushed successfully
    """
    try:
        if events_df is None or len(events_df) == 0:
            logger.info(f"No events for match {match_id} yet")
            return False

        # Filter out already processed events
        new_events_df = filter_new_events(match_id, events_df)
        
        if len(new_events_df) == 0:
            logger.info(f"No new events to push for match {match_id}")
            return True  # Not an error, just no new data

        # Enrich match_info with league name
        match_info["league_name"] = _get_league_name(match_info.get("competition_id"))
        
        # Convert events to JSON
        events_json = new_events_df.to_json(orient="records")
        
        # Prepare stream message
        message = {
            "match_id": str(match_id),
            "match_info": json.dumps(match_info),
            "events": events_json,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_count": str(len(new_events_df)),
            "total_events": str(len(events_df))
        }
        
        # Push to Redis Stream
        stream_id = r.xadd(STREAM_NAME, message, maxlen=10000)  # Keep last 10k messages
        
        # Mark events as processed
        mark_events_as_processed(match_id, new_events_df)
        
        logger.info(f"✓ Pushed {len(new_events_df)} new events for match {match_id} to stream (ID: {stream_id.decode()})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push events for match {match_id}: {e}", exc_info=True)
        return False


def fetch_and_push_match(match_id: int, match_info: Dict) -> bool:
    """
    Fetch events for a match and push to stream
    
    Returns:
        bool: True if match is complete and should not be retried
    """
    try:
        logger.info(f"Fetching events for match {match_id}...")
        events = fetch_live_match_event(client, match_id)
        
        if events is None or len(events) == 0:
            logger.info(f"Match {match_id} has no events yet")
            return False
        
        # Push to stream (with deduplication)
        success = push_events_to_stream(match_id, match_info, events)
        
        if not success:
            return False
        
        # Check if match is finished
        is_finished = False
        if "status" in events.columns:
            is_finished = events["status"].astype(str).str.upper().eq("COMPLETE").any()
        elif "minute" in events.columns:
            is_finished = events["minute"].max() >= 90
        
        if is_finished:
            seen_matches.add(match_id)
            logger.info(f"✓ Match {match_id} is complete (marked as processed)")
            return True
        else:
            logger.info(f"↻ Match {match_id} is ongoing (will retry next cycle)")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching match {match_id}: {e}", exc_info=True)
        return False


def fetch_and_process_window():
    """Fetch matches in UTC rolling window and push events to stream"""
    utc_today = datetime.now(timezone.utc).date()
    min_date = (utc_today - timedelta(days=3)).strftime("%Y-%m-%d")
    max_date = (utc_today + timedelta(days=3)).strftime("%Y-%m-%d")
    
    logger.info(f"[Window Check] Fetching matches from {min_date} to {max_date}")
    
    matches = fetch_matches_by_date(client, min_date=min_date, max_date=max_date)
    
    if matches is None or len(matches) == 0:
        logger.info("No matches found in window")
        return
    
    # Only process matches up to today (not future matches)
    utc_today_str = utc_today.strftime("%Y-%m-%d")
    matches = matches[matches["match_date"] <= utc_today_str].reset_index(drop=True)
    
    if len(matches) == 0:
        logger.info("No processable matches (all are in the future)")
        return
    
    logger.info(f"Found {len(matches)} matches to check")
    
    for _, match in matches.iterrows():
        match_id = int(match["match_id"])
        
        # Skip if already completed
        if match_id in seen_matches:
            continue
        
        fetch_and_push_match(match_id, match.to_dict())
        time.sleep(1)  # Rate limiting


def listen_new_match_channel():
    """Listen for immediate match notifications"""
    pubsub = r.pubsub()
    pubsub.subscribe("new_match_channel")
    logger.info("Listening on 'new_match_channel' for immediate triggers...")
    
    for msg in pubsub.listen():
        if msg["type"] != "message":
            continue
        
        try:
            data = json.loads(msg["data"].decode("utf-8"))
            match_id = int(data.get("match_id"))
            match_info = data.get("match_info", {})
            
            if match_id not in seen_matches:
                logger.info(f"[New Match Trigger] Processing match {match_id}")
                fetch_and_push_match(match_id, match_info)
                
        except Exception as e:
            logger.error(f"Error handling new_match_channel: {e}", exc_info=True)


def cleanup_old_tracking_data():
    """Clean up old event tracking data to prevent memory bloat"""
    logger.info("Cleaning up old event tracking data...")
    
    # Find all processed_events and event_hashes keys
    cursor = 0
    cleaned_count = 0
    
    while True:
        cursor, keys = r.scan(cursor, match="processed_events:*", count=100)
        
        for key in keys:
            # Check if key has TTL, if not set it
            ttl = r.ttl(key)
            if ttl == -1:  # No TTL set
                r.expire(key, PROCESSED_EVENTS_TTL)
                cleaned_count += 1
        
        if cursor == 0:
            break
    
    # Same for event_hashes
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="event_hashes:*", count=100)
        
        for key in keys:
            ttl = r.ttl(key)
            if ttl == -1:
                r.expire(key, PROCESSED_EVENTS_TTL)
                cleaned_count += 1
        
        if cursor == 0:
            break
    
    if cleaned_count > 0:
        logger.info(f"Set TTL for {cleaned_count} tracking keys")


def main():
    logger.info("=" * 60)
    logger.info("Starting Event Producer v2 (Redis Streams + Deduplication)")
    logger.info(f"Stream: {STREAM_NAME}")
    logger.info(f"Deduplication TTL: {PROCESSED_EVENTS_TTL}s")
    logger.info("=" * 60)
    
    # Cleanup old tracking data on startup
    cleanup_old_tracking_data()
    
    # Start listener thread
    listener_thread = threading.Thread(target=listen_new_match_channel, daemon=True)
    listener_thread.start()
    
    # Polling loop
    check_interval = 300  # 5 minutes
    cleanup_interval = 3600  # 1 hour
    last_cleanup = time.time()
    
    try:
        while True:
            fetch_and_process_window()
            
            # Periodic cleanup
            if time.time() - last_cleanup > cleanup_interval:
                cleanup_old_tracking_data()
                last_cleanup = time.time()
            
            logger.info(f"Sleeping for {check_interval}s...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        logger.info("Shutting down producer...")


if __name__ == "__main__":
    main()