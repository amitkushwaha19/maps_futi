#!/usr/bin/env python3
"""
Event Consumer v2 - Processes events from Redis Stream with Deduplication
=========================================================================

Features:
- Consumes raw events from Redis Stream: 'live_events_stream'
- Processes through SPADL → VAEP pipeline
- Publishes results to Redis PubSub: 'vaep_output_channel'
- Appends results to CSV: 'vaep_output_data.csv'
- Uses consumer groups for scalability and fault tolerance
- DEDUPLICATION: Prevents reprocessing of already handled messages
"""

import os
import sys
import time
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import numpy as np
import joblib
import redis
from dotenv import load_dotenv

# Import processing functions
sys.path.append(os.path.abspath(os.path.join('.', 'functions')))
from functions import spadl_processing_functions
from vaep_functions import vaep_generate_features_all_games, prep_X

# Load environment
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [CONSUMER_V2] - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("consumer_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)

# Load ML Models
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "models")

logger.info("Loading VAEP models...")
model_scores = joblib.load(os.path.join(model_dir, "vaep-scores-model-phases.pkl"))
model_concedes = joblib.load(os.path.join(model_dir, "vaep-concedes-model-phases.pkl"))
feature_cols = joblib.load(os.path.join(model_dir, "vaep-feature-columns.pkl"))
logger.info("✓ Models loaded")

# Stream Configuration
STREAM_NAME = "live_events_stream"
CONSUMER_GROUP = "vaep_processors"
CONSUMER_NAME = f"consumer_{os.getpid()}"

# Deduplication settings
PROCESSED_MESSAGES_TTL = 86400 * 7  # 7 days TTL for processed messages tracking


# -------------------- Deduplication Functions --------------------

def get_processed_messages_key() -> str:
    """Get Redis key for tracking processed messages"""
    return f"processed_messages:{CONSUMER_GROUP}"


def generate_message_hash(match_id: int, events_json: str) -> str:
    """
    Generate hash for message content to detect duplicates
    """
    content = f"{match_id}:{events_json}"
    return hashlib.md5(content.encode()).hexdigest()


def is_message_processed(msg_id: str, match_id: int, events_json: str) -> bool:
    """
    Check if message has already been processed
    """
    processed_key = get_processed_messages_key()
    
    # Check if message ID exists
    if r.sismember(processed_key, msg_id):
        return True
    
    # Check content hash for duplicate content with different IDs
    content_hash = generate_message_hash(match_id, events_json)
    hash_key = f"message_hashes:{CONSUMER_GROUP}"
    
    if r.sismember(hash_key, content_hash):
        logger.info(f"Duplicate content detected for message {msg_id} (different ID, same content)")
        return True
    
    return False


def mark_message_as_processed(msg_id: str, match_id: int, events_json: str):
    """
    Mark message as processed
    """
    processed_key = get_processed_messages_key()
    hash_key = f"message_hashes:{CONSUMER_GROUP}"
    content_hash = generate_message_hash(match_id, events_json)
    
    # Use pipeline for atomic operations
    pipe = r.pipeline()
    pipe.sadd(processed_key, msg_id)
    pipe.sadd(hash_key, content_hash)
    pipe.expire(processed_key, PROCESSED_MESSAGES_TTL)
    pipe.expire(hash_key, PROCESSED_MESSAGES_TTL)
    pipe.execute()


# -------------------- Processing Functions --------------------

def _extract_positions(events: pd.DataFrame) -> pd.DataFrame:
    """Extract player positions from lineup data"""
    try:
        pos_map = {}
        if "lineup" in events.columns:
            for _, row in events.iterrows():
                lv = row["lineup"]
                if isinstance(lv, list):
                    for p in lv:
                        if isinstance(p, dict):
                            pid = p.get("player_id")
                            pos = p.get("position")
                            if pid and pos:
                                pos_map[pid] = pos
        if pos_map:
            events["position"] = events["player_id"].map(pos_map)
            events["gk_position"] = events["position"]
    except Exception as e:
        logger.warning(f"Position extraction failed: {e}")
    return events


def _format_to_spadl(events_df: pd.DataFrame) -> pd.DataFrame:
    """Convert StatsBomb events to SPADL format"""
    spadl_df = spadl_processing_functions.format_event_data(
        events_df, 
        in_format="statsbomb"
    ).reset_index(drop=True)
    
    # Preserve metadata
    for field in ["statsbomb_match_id", "statsbomb_static_id", "season_id", "statsbomb_team_id"]:
        if field in events_df.columns and field not in spadl_df.columns:
            if field == "statsbomb_team_id":
                team_map = events_df.groupby("team_id")[field].first().to_dict()
                spadl_df[field] = spadl_df["team_id"].map(team_map)
            else:
                spadl_df[field] = events_df[field].iloc[0]
    
    return spadl_df


def _compute_vaep(spadl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VAEP values for SPADL actions"""
    
    # Generate features
    feats = vaep_generate_features_all_games(spadl_df)
    feats = prep_X(feats, feature_cols, convert_to_sparse=True)
    
    # Predict probabilities
    spadl_df["prob_score"] = model_scores.predict_proba(feats)[:, 1]
    spadl_df["prob_concede"] = model_concedes.predict_proba(feats)[:, 1]

    # Restore original shot info
    shot_rows = spadl_df["type_name"].isin(["shot", "shot_freekick", "shot_penalty"])
    if "shot_original_result_id" in spadl_df.columns:
        spadl_df.loc[shot_rows, "result_id"] = spadl_df.loc[shot_rows, "shot_original_result_id"].astype("float64")
    if "shot_original_result_name" in spadl_df.columns:
        spadl_df.loc[shot_rows, "result_name"] = spadl_df.loc[shot_rows, "shot_original_result_name"]
    if "shot_original_x_end" in spadl_df.columns:
        spadl_df.loc[shot_rows, "x_end"] = spadl_df.loc[shot_rows, "shot_original_x_end"]
    if "shot_original_y_end" in spadl_df.columns:
        spadl_df.loc[shot_rows, "y_end"] = spadl_df.loc[shot_rows, "shot_original_y_end"]
    
    # Drop temporary columns
    spadl_df.drop(
        columns=[c for c in ["shot_original_result_id", "shot_original_result_name", 
                             "shot_original_x_end", "shot_original_y_end"] 
                 if c in spadl_df.columns], 
        inplace=True, 
        errors="ignore"
    )

    # Calculate previous action scores
    spadl_df["prev_scores"] = np.where(
        (spadl_df["team_id"] == spadl_df["team_id"].shift()) &
        (spadl_df["period_id"] == spadl_df["period_id"].shift()) &
        (spadl_df["match_id"] == spadl_df["match_id"].shift()),
        spadl_df["prob_score"].shift(),
        spadl_df["prob_concede"].shift()
    )
    
    spadl_df["prev_time"] = spadl_df.groupby(["match_id", "period_id"])["time_seconds"].shift()

    # Handle own goals
    spadl_df["goal"] = np.where(spadl_df["result_name"] == "owngoal", 1, spadl_df["goal"])
    
    # Adjust prev_scores based on context
    spadl_df["prev_scores"] = np.where(spadl_df["goal"].shift() == 1, 0, spadl_df["prev_scores"])
    spadl_df["prev_scores"] = np.where(spadl_df["time_seconds"] - spadl_df["prev_time"] > 10, 0, spadl_df["prev_scores"])
    spadl_df["prev_scores"] = np.where(spadl_df["type_name"] == "shot_penalty", 0.792453, spadl_df["prev_scores"])
    spadl_df["prev_scores"] = np.where(spadl_df["type_name"].isin(["corner_crossed", "corner_short"]), 0.0465, spadl_df["prev_scores"])
    spadl_df["prev_scores"] = np.where(spadl_df["type_name"] == "goalkick", 0, spadl_df["prev_scores"])
    spadl_df["prev_scores"] = spadl_df["prev_scores"].fillna(0)

    # Calculate offensive VAEP
    spadl_df["offensive_vaep"] = spadl_df["prob_score"] - spadl_df["prev_scores"]

    # Calculate previous concedes
    spadl_df["prev_concedes"] = np.where(
        (spadl_df["team_id"] == spadl_df["team_id"].shift()) &
        (spadl_df["period_id"] == spadl_df["period_id"].shift()) &
        (spadl_df["match_id"] == spadl_df["match_id"].shift()),
        spadl_df["prob_concede"].shift(),
        spadl_df["prob_score"].shift()
    )
    spadl_df["prev_concedes"] = np.where(spadl_df["goal"].shift() == 1, 0, spadl_df["prev_concedes"])
    spadl_df["prev_concedes"] = np.where(spadl_df["type_name"] == "goalkick", 0, spadl_df["prev_concedes"])
    spadl_df["prev_concedes"] = np.where(spadl_df["time_seconds"] - spadl_df["prev_time"] > 10, 0, spadl_df["prev_concedes"])
    spadl_df["prev_concedes"] = spadl_df["prev_concedes"].fillna(0)

    # Calculate defensive VAEP
    spadl_df["defensive_vaep"] = spadl_df["prev_concedes"] - spadl_df["prob_concede"]

    # Total VAEP
    spadl_df["vaep"] = spadl_df["offensive_vaep"] + spadl_df["defensive_vaep"]
    spadl_df["vaep"] = np.where(spadl_df["type_name"] == "shot_penalty", 0, spadl_df["vaep"])

    # Split passes into pass + reception
    other = spadl_df[~spadl_df["type_name"].isin(
        ["pass", "freekick_short", "freekick_crossed", "corner_short", "corner_crossed", "throw_in", "cross"]
    )]
    
    passes = spadl_df[spadl_df["type_name"].isin(
        ["pass", "freekick_short", "freekick_crossed", "corner_short", "corner_crossed", "throw_in", "cross"]
    )].copy()
    
    passes["vaep"] = passes["vaep"] / 2
    
    # Create reception actions
    recs = passes.copy()
    recs["player_id"] = recs["receiver_id"]
    recs["receiver_id"] = np.nan
    recs["type_name"] = "reception"
    recs["type_id"] = 99
    recs["bodypart_id"] = np.nan
    recs["bodypart_name"] = ""
    recs["x_start"] = recs["x_end"]
    recs["y_start"] = recs["y_end"]
    recs["action_id"] = recs["action_id"] + 0.5

    # Combine all actions
    allocated = pd.concat([other, passes, recs]).sort_values(
        by=["match_id", "period_id", "action_id"]
    ).reset_index(drop=True)
    
    # Remove failed receptions
    allocated = allocated[
        ~((allocated["type_name"] == "reception") & (allocated["result_name"] != "success"))
    ].reset_index(drop=True)
    
    # No negative VAEP for receptions
    allocated.loc[(allocated["type_name"] == "reception") & (allocated["vaep"] < 0), "vaep"] = 0

    # Add gamestate
    try:
        allocated = spadl_processing_functions.add_gamestate(allocated)
    except Exception as e:
        logger.warning(f"Gamestate not added: {e}")

    return allocated


def process_event_message(match_id: int, match_info: Dict, events_json: str):
    """Process a single event message from the stream"""
    try:
        # Parse events
        events = pd.read_json(events_json, orient="records")
        
        if len(events) == 0:
            logger.warning(f"Empty events for match {match_id}")
            return
        
        logger.info(f"Processing {len(events)} events for match {match_id}")
        
        # Add metadata to events
        events["match_date"] = match_info.get("match_date")
        events["match_home_team_name"] = match_info.get("match_home_team_name")
        events["match_away_team_name"] = match_info.get("match_away_team_name")
        events["competition_id"] = match_info.get("competition_id")
        events["season_id"] = match_info.get("season_id")
        events["statsbomb_match_id"] = match_id

        # Determine home/away
        if "team_id" in events.columns and len(events["team_id"].dropna()) > 0:
            home_team_id = events["team_id"].mode().iloc[0]
            events["home"] = events["team_id"] == home_team_id
            events["away"] = events["team_id"] != home_team_id
            events["home"] = events["home"].map({True: "home", False: "away"})
            events["away"] = events["away"].map({True: "away", False: "home"})

        events["team_name"] = events.apply(
            lambda row: match_info["match_home_team_name"] if row.get("home") == "home"
            else match_info["match_away_team_name"], axis=1
        )
        events["league_name"] = match_info.get("league_name", "Unknown")

        # Calculate elapsed time
        if {"minute", "second"}.issubset(events.columns):
            events["elapsed_time"] = events["minute"] * 60 + events["second"]

        # Extract positions
        events = _extract_positions(events)

        # Ensure numeric types
        numeric_cols = [
            "x_start", "y_start", "x_end", "y_end", "time_seconds",
            "action_id", "result_id", "bodypart_id", "type_id"
        ]
        for col in numeric_cols:
            if col in events.columns:
                events[col] = pd.to_numeric(events[col], errors="coerce")

        # Drop rows with missing critical data
        required_cols = ["x_start", "y_start", "time_seconds"]
        existing_cols = [c for c in required_cols if c in events.columns]
        if existing_cols:
            events.dropna(subset=existing_cols, inplace=True)

        # Convert to SPADL
        spadl_df = _format_to_spadl(events)
        if "match_id" not in spadl_df.columns:
            spadl_df["match_id"] = match_id

        # Compute VAEP
        vaep_df = _compute_vaep(spadl_df)

        # Add match metadata
        vaep_df["match_id"] = match_id
        vaep_df["match_home_team_name"] = match_info.get("match_home_team_name")
        vaep_df["match_away_team_name"] = match_info.get("match_away_team_name")
        vaep_df["match_date"] = match_info.get("match_date")
        vaep_df["competition_id"] = match_info.get("competition_id")
        vaep_df["season_id"] = match_info.get("season_id")

        # Publish to Redis PubSub
        out_json = vaep_df.to_json(orient="records")
        r.publish("vaep_output_channel", out_json)
        r.set(f"vaep_match_{match_id}", out_json)

        # Append to CSV
        csv_path = os.path.join(current_dir, "vaep_output_data.csv")
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        vaep_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

        logger.info(f"✓ Published VAEP for match {match_id} ({len(vaep_df)} actions)")

    except Exception as e:
        logger.error(f"Error processing match {match_id}: {e}", exc_info=True)
        raise  # Re-raise to mark message as failed


def create_consumer_group():
    """Create consumer group if it doesn't exist"""
    try:
        r.xgroup_create(STREAM_NAME, CONSUMER_GROUP, id='0', mkstream=True)
        logger.info(f"✓ Created consumer group '{CONSUMER_GROUP}'")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"Consumer group '{CONSUMER_GROUP}' already exists")
        else:
            raise


def cleanup_old_tracking_data():
    """Clean up old message tracking data"""
    logger.info("Cleaning up old message tracking data...")
    
    processed_key = get_processed_messages_key()
    hash_key = f"message_hashes:{CONSUMER_GROUP}"
    
    # Set TTL if not already set
    if r.ttl(processed_key) == -1:
        r.expire(processed_key, PROCESSED_MESSAGES_TTL)
    if r.ttl(hash_key) == -1:
        r.expire(hash_key, PROCESSED_MESSAGES_TTL)


def consume_stream():
    """Consume messages from Redis Stream"""
    logger.info(f"Starting consumer: {CONSUMER_NAME}")
    logger.info(f"Reading from stream: {STREAM_NAME}")
    logger.info(f"Consumer group: {CONSUMER_GROUP}")
    
    while True:
        try:
            # Read messages (block for 5 seconds)
            messages = r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_NAME: '>'},
                count=1,
                block=5000
            )
            
            if not messages:
                continue
            
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    msg_id_str = msg_id.decode()
                    
                    try:
                        # Decode message
                        match_id = int(msg_data[b"match_id"].decode())
                        match_info = json.loads(msg_data[b"match_info"].decode())
                        events_json = msg_data[b"events"].decode()
                        
                        logger.info(f"[{msg_id_str}] Processing match {match_id}")
                        
                        # Check for duplicates
                        if is_message_processed(msg_id_str, match_id, events_json):
                            logger.info(f"[{msg_id_str}] Message already processed, skipping")
                            # Still ACK to remove from pending
                            r.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                            continue
                        
                        # Process the message
                        process_event_message(match_id, match_info, events_json)
                        
                        # Mark as processed
                        mark_message_as_processed(msg_id_str, match_id, events_json)
                        
                        # Acknowledge successful processing
                        r.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)
                        logger.info(f"✓ ACK message {msg_id_str}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process message {msg_id_str}: {e}", exc_info=True)
                        # Message will be retried later
                        
        except Exception as e:
            logger.error(f"Stream read error: {e}", exc_info=True)
            time.sleep(5)


def main():
    logger.info("=" * 60)
    logger.info("Starting VAEP Consumer v2 (with Deduplication)")
    logger.info(f"Deduplication TTL: {PROCESSED_MESSAGES_TTL}s")
    logger.info("=" * 60)
    
    # Create consumer group
    create_consumer_group()
    
    # Cleanup old tracking data
    cleanup_old_tracking_data()
    
    # Start consuming
    try:
        consume_stream()
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")


if __name__ == "__main__":
    main()