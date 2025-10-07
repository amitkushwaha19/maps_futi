import pandas as pd
import requests
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from tqdm import tqdm

def fetch_access_token(client_id, client_secret):
    """
    Get the token for the StatsBomb API.

    Parameters:
        client_id: The client ID for the API.
        client_secret: The client secret for the API.
    """
    if not client_id or not client_secret:
        raise ValueError("Client ID and Client Secret are required")
    
    # Define the URL and payload
    url = "https://live-api.statsbomb.com/v1/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret
    }

    # Define the headers
    headers = {
        "content-type": "application/json"
    }

    # Make the POST request
    response = requests.post(url, json=payload, headers=headers)
    
    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to fetch access token: {response.status_code} - {response.text}")
    
    try:
        response_data = response.json()
    except requests.exceptions.JSONDecodeError:
        raise Exception(f"Invalid JSON response: {response.text}")
    
    access_token = response_data.get("access_token")
    if not access_token:
        raise Exception(f"No access token in response: {response_data}")
    
    return access_token

def create_sblive_client(access_token):
    """
    Create a GraphQL client for the StatsBomb API.

    Parameters:
        access_token: The access token for the API.

    Returns:
        A GraphQL client instance.
    """
    # Define the transport layer
    transport = RequestsHTTPTransport(
        url="https://live-api.statsbomb.com/v1/graphql",
        headers={"Authorization": f"Bearer {access_token}"},
        use_json=True
    )

    # Create the client
    client = Client(transport=transport, fetch_schema_from_transport=False)

    return client

def fetch_all_competitions(client):
    """
    Fetch competition info for all available competitions.

    Parameters:
        client: The GraphQL client instance.

    Returns:
        A pandas DataFrame containing all available competitions.
    """
    query = gql(
        """
        query Live_match($where: live_competition_season_bool_exp) {
          live_competition_season(where: $where) {
            season_name
            competition_name
            competition_id
            season_id
            }
        }
        """
    )

    result = client.execute(query)
    competitions = pd.DataFrame(result['live_competition_season'])

    return competitions

def fetch_matches_by_date(client, min_date=None, max_date=None, season_id=None):
    """
    Fetch match info for matches within a specific timeframe and/or season(s).

    Parameters:
        client: The GraphQL client instance.
        min_date: The start date of the timeframe (yyyy-mm-dd format). Optional.
        max_date: The end date of the timeframe (yyyy-mm-dd format). Optional.
        season_id: A single season_id or a list of season_ids. Optional.

    Returns:
        A pandas DataFrame containing all available matches based on the filters.
    """
    query = gql(
        """
        query Live_match($where: live_match_bool_exp) {
          live_match(where: $where) {
            competition_id
            match_id
            match_away_score
            match_away_team_name
            match_date
            match_home_score
            match_home_team_name
            match_name
            season_id
            match_neutral_ground
            round_id
            round_type_name
          }
        }
        """
    )

    # Build the "where" clause dynamically based on non-None inputs
    where_clause = {}
    if min_date and max_date:
        where_clause["match_date"] = {"_gt": min_date, "_lt": max_date}
    elif min_date:
        where_clause["match_date"] = {"_gt": min_date}
    elif max_date:
        where_clause["match_date"] = {"_lt": max_date}

    if season_id:
        if isinstance(season_id, list):
            where_clause["season_id"] = {"_in": season_id}
        else:
            where_clause["season_id"] = {"_eq": season_id}

    params = {"where": where_clause}

    result = client.execute(query, variable_values=params)
    matches_data = pd.DataFrame(result['live_match'])
    matches_data = matches_data.sort_values(by='match_date')
    matches_data = matches_data.reset_index(drop=True)

    return matches_data

def fetch_live_match_event(client, match):
    """
    Fetch live match event data for a specific match ID.

    Parameters:
        client: The GraphQL client instance.
        match: The match ID to query.

    Returns:
        A pandas DataFrame containing the live match event data.
    """
    query = gql(
        """
        query Live_match_event($where: live_match_event_bool_exp) {
          live_match_event(where: $where) {
            advantage
            aerial_won
            assist
            backheel
            body_part
            card
            competition_id
            created_at
            defensive
            deflection
            distance_to_opponents_goal
            distance_to_own_goal
            duration
            end_x
            end_y
            end_z
            finished
            first_time
            formation
            freeze_frame
            from_corner
            from_free_kick
            from_open_play
            from_penalty_box
            from_set_piece
            from_six_yard_box
            from_throw_in
            goal_against
            goal_for
            height
            id
            in_attacking_half
            in_defensive_half
            in_defensive_third
            in_final_third
            in_penalty_box
            in_six_yard_box
            index
            inside_attacking_half
            inside_defensive_half
            inside_defensive_third
            inside_final_third
            inside_penalty_box
            inside_six_yard_box
            into_attacking_half
            into_defensive_half
            into_defensive_third
            into_final_third
            into_penalty_box
            into_six_yard_box
            key_pass
            lineup
            match_id
            minute
            name
            next_period
            no_touch
            nutmeg
            off_camera
            offensive
            opposition_id
            outcome
            overrun
            penalty
            period
            player_id
            position
            recipient_id
            recovery_failure
            replacement_id
            save_block
            season_id
            second
            start_x
            start_y
            start_z
            status
            team_id
            technique
            timestamp
            type
            updated_at
            xg
          }
        }
        """
    )

    params = {
        "where": {
            "match_id": {
                "_eq": match  # Match ID passed as a parameter
            }
        }
    }

    # Execute the query
    result = client.execute(query, variable_values=params)

    # Convert the result to a pandas DataFrame
    df = pd.DataFrame(result['live_match_event'])
    df = df.reset_index(drop=True)

    return df

def fetch_all_live_match_events(client, matches):
    """
    Fetch live match event data for multiple match IDs and combine results into a single DataFrame.

    Parameters:
        client: The GraphQL client instance.
        matches: A list of match IDs to query.

    Returns:
        A pandas DataFrame containing the combined live match event data for all match IDs.
    """
    all_results = []

    for match in tqdm(matches, desc="Fetching match data", unit="match"):
        df = fetch_live_match_event(client, match)  # Call the function for each match ID
        all_results.append(df)  # Append the result to the list

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)

    return combined_df

def fetch_player_info(client, player_ids):
    """
    Fetch player details for a specific player ID or list of player IDs.

    Parameters:
        client: The GraphQL client instance.
        player_id: The player ID to query.

    Returns:
        A pandas DataFrame containing the player details.
    """
    query = gql(
        """
        query PlayerDetailsByFullName($player_ids: [Int!]!) {
          live_lineup(
            where: { 
              player_id: { _in: $player_ids }
            },
            distinct_on: player_id
          ) {
            player_id
            player_name
            player_nationality
            player_nickname
            player_preferred_foot
            player_weight
            player_height
            player_date_of_birth
            player_country_of_birth
            player_firstname
            player_lastname
          }
        }
        """
    )

    params = {
        "player_ids": player_ids  # Player ID passed as a parameter
    }

    # Execute the query
    result = client.execute(query, variable_values=params)

    # Convert the result to a pandas DataFrame
    df = pd.DataFrame(result['live_lineup'])
    df = df.reset_index(drop=True)

    return df

def fetch_team_info(client, team_ids):
    """
    Fetch team details for a specific team ID or list of team IDs.

    Parameters:
        client: The GraphQL client instance.
        team_id: The team ID or list of team IDs to query.

    Returns:
        A pandas DataFrame containing the player details.
    """
    query = gql(
        """
        query PlayerDetailsByFullName($team_ids: [Int!]!) {
          live_lineup(
            where: { 
              team_id: { _in: $team_ids }
            },
            distinct_on: team_id
          ) {
            team_id
            team_name
          }
        }
        """
    )

    params = {
        "team_ids": team_ids  # Player ID passed as a parameter
    }

    # Execute the query
    result = client.execute(query, variable_values=params)

    # Convert the result to a pandas DataFrame
    df = pd.DataFrame(result['live_lineup'])
    df = df.reset_index(drop=True)

    return df

def fetch_player_id_map(player_ids, sb_static_username, sb_static_password):
    """
    Fetch a mapping of statsbomb live to statsbomb static player IDs.
    Uses the statsbomb static API, so separate from the live API client with different credentials.
    The endpoint returns a different row for every player/team/season combination, so it can be repurposed for teams and seasons too if needed.

    Parameters:
        player_ids: A list of statsbomb live IDs to query.

    Returns:
        A pandas DataFrame containing the statsbomb live player IDs and their corresponding static player IDs.
    """
    all_players = []
    for player_id in player_ids:
        url = "https://data.statsbomb.com/api/v1/player-mapping?&live-player-id=" + str(int(player_id)) + "&add-matches-played=false"
        response = requests.get(url, auth=(sb_static_username, sb_static_password))
        if response.status_code != 200:
            raise Exception(f"Failed to fetch player ID mapping for {player_id}: {response.status_code} - {response.text}")
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df = df[['offline_player_id', 'live_player_id']].drop_duplicates().reset_index(drop=True)
            df.rename(columns={'offline_player_id': 'statsbomb_static_id', 'live_player_id': 'statsbomb_live_id'}, inplace=True)
            all_players.append(df)
    if all_players:
        return pd.concat(all_players, ignore_index=True)
    else:
        return pd.DataFrame(columns=['statsbomb_static_id', 'statsbomb_live_id'])