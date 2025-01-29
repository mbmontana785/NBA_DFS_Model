import pandas as pd
import sqlite3
import requests
from datetime import datetime, timedelta
import streamlit as st
from functions import set_yesterday_only, fetch_game_ids, get_row_count, update_game_stats, check_missing_values, fetch_player_salaries, populate_position_columns

today_df = None

# Display today's date
today = datetime.now().strftime("%Y%m%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
last_day = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
formatted_date = datetime.now().strftime("%B %d, %Y")

# Streamlit App
st.title(f"NBA DFS Model and Lineup Generator for {formatted_date}")

# API key for fetching data
api_key = "3103a75392msh7bce7c32fde122cp134393jsn4d42ed6d08a8"

# Determine yesterday_only status
yesterday_only = set_yesterday_only(yesterday, last_day)

if yesterday_only is None:
    #st.write("Game data is already up-to-date through yesterday.")
    game_ids = []
elif yesterday_only:
    st.write("Fetching game data for yesterday only.")
    game_ids, no_game_dates = fetch_game_ids(api_key, yesterday, last_day, today, yesterday_only=True)
else:
    st.write("Fetching game data for multiple days.")
    game_ids, no_game_dates = fetch_game_ids(api_key, yesterday, last_day, today, yesterday_only=False)

# Display game IDs
if game_ids:
    st.subheader("Game IDs to Process")
    st.write(game_ids)
else:
    st.warning("No new games to process.")

# Get and display row count before update
initial_row_count = get_row_count()
st.write(f"Rows in game_stats table before update: {initial_row_count}")

if yesterday_only is not None:  # Only show the button if there are games to update
    if st.button("Update Database with Box Scores"):
        update_game_stats(api_key, game_ids)
        
        # Get and display row count after update
        final_row_count = get_row_count()
        st.write(f"Rows in game_stats table after update: {final_row_count}")

# Connect to the database and load the entire table into main_df
conn = sqlite3.connect("nba_dfs_model.db")  # Adjust the path as needed
main_df = pd.read_sql_query("SELECT * FROM game_stats", conn)
conn.close()

# Display the last five rows of the table
st.subheader("Last 5 Rows of game_stats Table")
st.dataframe(main_df.tail())  # Show the last 5 rows in an interactive table

#Dropping identical rows, if any (there shouldn't be any)
main_df = main_df.drop_duplicates()

#Deriving a date column, and dropping rows with duplicate player_ids and games
main_df['date'] = pd.to_datetime(main_df['game_id'].str[:8])
main_df = main_df.drop_duplicates(subset=['longName', 'player_id', 'team', 'date'], keep='first')

#Filtering out anything before Jan. 1, 2024, to cut down on missing values
main_df = main_df[main_df['date'] >= '2024-01-01']
#Identifying numeric columns
num_cols = ['fga', 'ast', 'tptfgm', 'fgm', 'fta', 'tptfga', 'OffReb', 'ftm', 'blk', 'DefReb', 'plusMinus', 'stl', 'pts', 'PF', 'TOV', 'usage', 'mins']

#Changing numeric columns to numeric type
main_df[num_cols] = main_df[num_cols].apply(pd.to_numeric, errors = 'coerce')
#Sorting the dataframe by player_id, date, and game_id so all player rows are together in order of games played
main_df_sorted = main_df.sort_values(by = ['player_id', 'date', 'game_id']).reset_index(drop = True)

#Checking for missing values
check_missing_values(main_df_sorted)
    
# Initialize session state for site selection
if 'site' not in st.session_state:
    st.session_state.site = None

# Display buttons for FanDuel and DraftKings selection
col1, col2 = st.columns(2)

with col1:
    if st.button("FanDuel"):
        st.session_state.site = "FanDuel"

with col2:
    if st.button("DraftKings"):
        st.session_state.site = "DraftKings"

# Show the selected site and positions
if st.session_state.site:
    st.write(f"You selected: {st.session_state.site}")
    
    if st.session_state.site == "FanDuel":
        positions = ["PG", "SG", "SF", "PF", "C"]
    else:
        positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    # Calling the API to get player salaries for that night's slate of games
    today_df = fetch_player_salaries(api_key, today, st.session_state.site.lower(), positions)

    if today_df is not None:
        # Populate binary position columns
        today_df = populate_position_columns(today_df, positions, st.session_state.site.lower())
        #Processing today_df before next api call
        today_df = today_df.drop(columns = ['pos'])
        today_df['date'] = pd.to_datetime(today)
        today_df['game_id'] = ''

else:
    st.warning("Please select a site to proceed.")


# Getting game_ids for tonight's games
game_ids, no_game_dates = fetch_game_ids(api_key, date=today)

# Display game IDs
if game_ids:
    st.subheader("Today's game IDs")
    st.write(game_ids)
else:
    st.warning("No game_ids found for today.")

#Checking for incorrect game_ids
from collections import Counter

# List to store teams
teams = []

# Extract both teams from each game_id
for game_id in game_ids:
    matchup = game_id[9:]  # Assuming game_id format includes matchup info after index 9
    both_teams = matchup.split('@')
    teams.append(both_teams[0])
    teams.append(both_teams[1])

# Check for duplicates
if len(teams) != len(set(teams)):
    counter = Counter(teams)
    duplicates = [team for team, count in counter.items() if count > 1]

    # Display warning and list duplicates
    st.warning(f"These teams are in multiple game_ids: {duplicates}")

    # Show game_ids with their indices
    st.subheader("Remove game_ids")
    st.write("Below are the game_ids with duplicate teams:")
    for idx, game_id in enumerate(game_ids):
        st.write(f"{idx}: {game_id}")

    # Get user input for index
    game_id_index = st.number_input("Enter the index of the game_id to remove:", min_value=0, max_value=len(game_ids)-1, step=1, format="%d")

    if st.button("Remove game_id"):
        if 0 <= game_id_index < len(game_ids):
            removed_game_id = game_ids.pop(game_id_index)
            st.success(f"Removed game_id: {removed_game_id}")
        else:
            st.error("Invalid index. Please select a valid index.")