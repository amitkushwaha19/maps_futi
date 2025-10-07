#!/usr/bin/env python3
"""
Event Producer - Pushes raw StatsBomb events to Redis Streams
================================================================

Features:
- Fetches matches in UTC rolling window [yesterday .. tomorrow+3]
- Polls live events via GraphQL API
- Pushes raw events to Redis Stream: 'live_events_stream'
- Each message contains: match_id, match_info, raw_events (JSON)
- Listens to 'new_match_channel' for immediate triggers
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

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
    format="%(asctime)s - [PRODUCER] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("producer.log"),
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


def _get_league_name(competition_id: int) -> str:
    """Map competition ID to league name"""
    mapping = {
        44: "MLS", 37: "Premier League", 2: "Champions League",
        11: "La Liga", 9: "Serie A", 8: "Bundesliga", 4: "Ligue 1"
    }
    return mapping.get(int(competition_id) if competition_id is not None else competition_id,
                       f"Competition_{competition_id}")


def push_events_to_stream(match_id: int, match_info: Dict, events_df: pd.DataFrame) -> bool:
    """
    Push raw events to Redis Stream
    
    Returns:
        bool: True if events were pushed successfully
    """
    try:
        if events_df is None or len(events_df) == 0:
            logger.info(f"No events for match {match_id} yet")
            return False

        # Enrich match_info with league name
        match_info["league_name"] = _get_league_name(match_info.get("competition_id"))
        
        # Convert events to JSON
        events_json = events_df.to_json(orient="records")
        
        # Prepare stream message
        message = {
            "match_id": str(match_id),
            "match_info": json.dumps(match_info),
            "events": events_json,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Push to Redis Stream
        stream_id = r.xadd(STREAM_NAME, message, maxlen=10000)  # Keep last 10k messages
        
        logger.info(f"✓ Pushed {len(events_df)} events for match {match_id} to stream (ID: {stream_id.decode()})")
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
        
        # Push to stream
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


def main():
    logger.info("=" * 60)
    logger.info("Starting Event Producer (Redis Streams)")
    logger.info(f"Stream: {STREAM_NAME}")
    logger.info("=" * 60)
    
    # Start listener thread
    listener_thread = threading.Thread(target=listen_new_match_channel, daemon=True)
    listener_thread.start()
    
    # Polling loop
    check_interval = 300  # 5 minutes
    
    try:
        while True:
            fetch_and_process_window()
            logger.info(f"Sleeping for {check_interval}s...")
            time.sleep(check_interval)
    except KeyboardInterrupt:
        logger.info("Shutting down producer...")


if __name__ == "__main__":
    main()