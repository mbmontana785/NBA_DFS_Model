import pandas as pd
import numpy as np
import sqlite3
import requests
import pickle
import papermill
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt

# Import custom functions from functions.py
from functions import (
    set_yesterday_only, 
    fetch_game_ids, 
    get_row_count, 
    update_game_stats, 
    check_missing_values, 
    fetch_player_salaries, 
    populate_position_columns, 
    update_main_df,
    calculate_rolling_averages,
    fetch_injury_report,
    highlight_day_to_day
)


# API key for fetching data
api_key = "3103a75392msh7bce7c32fde122cp134393jsn4d42ed6d08a8"
api_host = "tank01-fantasy-stats.p.rapidapi.com"

# Display today's date
today = datetime.now().strftime("%Y%m%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
last_day = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
formatted_date = datetime.now().strftime("%B %#d, %Y")

# Streamlit App
st.title(f"NBA DFS Model and Lineup Generator for {formatted_date}")

# st.write("Session state:")

# st.session_state

# Initialize session state for site selection
if 'site' not in st.session_state:
    st.session_state.site = None

if 'positions' not in st.session_state:
    st.session_state.positions = None

if 'slate_df' not in st.session_state:
    st.session_state.slate_df = None

if 'teams_in_slate' not in st.session_state:
    st.session_state.teams_in_slate = None

if 'main_df' not in st.session_state:
    st.session_state.main_df = None

if 'today_df' not in st.session_state:
    st.session_state.today_df = pd.DataFrame()

if 'team_game_mins' not in st.session_state:
    st.session_state.team_game_mins = None

if "injury_df" not in st.session_state:
    st.session_state["injury_df"] = pd.DataFrame()

if "removed_players_df" not in st.session_state:
    st.session_state["removed_players_df"] = pd.DataFrame()  

# Display buttons for FanDuel and DraftKings selection
st.write("Do you want to play FanDuel or DraftKings?")

col1, col2 = st.columns(2)

with col1:
    if st.button("FanDuel"):
        st.session_state.site = "FanDuel"

with col2:
    if st.button("DraftKings"):
        st.session_state.site = "DraftKings"

if st.session_state.site:
    st.write(f"You selected {st.session_state.site}")

# Uploading slate file from computer
uploaded_file = st.file_uploader(f"Upload the {st.session_state.site} Player List", type=["csv"])

if uploaded_file is not None:
    st.session_state.slate_df = pd.read_csv(uploaded_file)  # Read the uploaded file
    if st.session_state.site == 'DraftKings':
        #DraftKings doesn't have an injury indicator
        #If we play DK we might just have to check the site for injuries
        dk_df = slate_df.rename(columns = {'TeamAbbrev': 'Team'})
        #Matching team names with API
        dk_df['Team'] = dk_df['Team'].replace({'SAS': 'SA', 'NOP': 'NO', 'NYK': 'NY'})
        st.session_state.teams_in_slate = set(list(dk_df['Team']))
    else:
        st.session_state.slate_df = st.session_state.slate_df[['Id', 'Position', 'Nickname', 'Team', 'Opponent', 'FPPG', 'Injury Indicator', 'Injury Details']]
        st.session_state.teams_in_slate = set(list(st.session_state.slate_df['Team']))
        
    st.success(f"{st.session_state.site} csv uploaded successfully!")

#Checking yesterday_only status, to see how many days of previous games need to be added to the game_stats table
yesterday_only, last_day = set_yesterday_only(yesterday, last_day)

if yesterday_only is None:
    game_ids = []
elif yesterday_only:
    st.write("Fetching game data for yesterday only.")
    game_ids, no_game_dates = fetch_game_ids(api_key, today, last_day, yesterday, yesterday_only=True)
else:
    st.write("Fetching game data for multiple days.")
    game_ids, no_game_dates = fetch_game_ids(api_key, today, last_day, yesterday, yesterday_only=False)

# Display game IDs
if game_ids:
    st.subheader("Game IDs to Process")
    st.write(game_ids)
else:
    st.warning("No new games to process.")

if yesterday_only is not None:  # Only show the button if there are games to update
    # Get and display row count before update
    initial_row_count = get_row_count()
    st.write(f"Rows in game_stats table before update: {initial_row_count}")
    if st.button("Update Database with Box Scores"):
        update_game_stats(api_key, game_ids)
        
        # Get and display row count after update
        final_row_count = get_row_count()
        st.write(f"Rows in game_stats table after update: {final_row_count}")

# Read data from the database, renaming 'PF' as 'fouls' so 'PF' can stand for power forward
conn = sqlite3.connect("nba_dfs_model.db")
st.session_state.main_df = pd.read_sql_query("SELECT *, PF AS fouls FROM game_stats", conn)
conn.close()

if st.button("Check last 5 rows of game_stats table?"):

    # Display the last five rows of the table
    st.subheader("Last 5 rows of game_stats table")
    st.dataframe(st.session_state.main_df.tail())  # Show the last 5 rows in an interactive table

#Dropping identical rows, if any (there shouldn't be any)
st.session_state.main_df = st.session_state.main_df.drop_duplicates()

#Deriving a date column, and dropping rows with duplicate player_ids and games
st.session_state.main_df['date'] = pd.to_datetime(st.session_state.main_df['game_id'].str[:8])
st.session_state.main_df = st.session_state.main_df.drop_duplicates(subset=['longName', 'player_id', 'team', 'date'], keep='first')

#Filtering out anything before Jan. 1, 2024, to cut down on missing values
st.session_state.main_df = st.session_state.main_df[st.session_state.main_df['date'] >= '2024-01-01']
#Identifying numeric columns
num_cols = ['fga', 'ast', 'tptfgm', 'fgm', 'fta', 'tptfga', 'OffReb', 'ftm', 'blk', 'DefReb', 'plusMinus', 'stl', 'pts', 'fouls', 'TOV', 'usage', 'mins']

#Changing numeric columns to numeric type
st.session_state.main_df[num_cols] = st.session_state.main_df[num_cols].apply(pd.to_numeric, errors = 'coerce')
#Sorting the dataframe by player_id, date, and game_id so all player rows are together in order of games played
st.session_state.main_df = st.session_state.main_df.sort_values(by = ['player_id', 'date', 'game_id']).reset_index(drop = True)

#Checking for missing values
check_missing_values(st.session_state.main_df)

if st.session_state.site:
    if st.session_state.site == "FanDuel":
        st.session_state.positions = ["PG", "SG", "SF", "PF", "C"]
    else:
        st.session_state.positions = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    # Calling the API to get player salaries for that night's slate of games
    st.session_state.today_df = fetch_player_salaries(api_key, today, st.session_state.site.lower(), st.session_state.positions)

    if st.session_state.today_df is not None:
        # Populate binary position columns
        st.session_state.today_df = populate_position_columns(st.session_state.today_df, st.session_state.positions, st.session_state.site.lower())
        #Processing today_df before next api call
        st.session_state.today_df = st.session_state.today_df.drop(columns = ['pos'])
        st.session_state.today_df['date'] = pd.to_datetime(today)
        st.session_state.today_df['game_id'] = ''

else:
    st.warning("Please select a site to proceed.")

# Getting game_ids for tonight's games
game_ids, no_game_dates = fetch_game_ids(api_key, today=today)

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

# Filling game_id column. Create a mapping from teams to game_ids
team_to_game_id = {team: game_id for game_id in game_ids for team in game_id[9:].split('@')}

# Fill in the game_id column based on the team
if not st.session_state['today_df'].empty:
    st.session_state.today_df['game_id'] = st.session_state.today_df['team'].map(team_to_game_id).fillna(st.session_state.today_df['game_id'])
    missing_values = check_missing_values(st.session_state.today_df)
    if missing_values is None:
        #Concatenating today_df with main_df
        st.session_state.main_df = update_main_df(st.session_state.main_df, st.session_state.today_df)
        st.session_state.main_df = st.session_state.main_df.sort_values(by = ['player_id', 'date', 'game_id']).reset_index(drop = True)

#Getting rolling means for last 15 games for every player, then sorting again just in case
if not st.session_state['main_df'].empty:
    st.session_state.main_df = calculate_rolling_averages(st.session_state.main_df, num_cols)
    st.session_state.main_df = st.session_state.main_df.sort_values(by=['player_id', 'date', 'game_id']).reset_index(drop=True)


# Identify rows with missing numeric values
rows_with_missing_values = st.session_state.main_df[st.session_state.main_df[num_cols].isnull().any(axis=1)]

# Count occurrences of each player_id in missing data
value_counts = rows_with_missing_values['player_id'].value_counts()

# Check if any player_id appears more than once
if not (value_counts == 1).all():
    st.warning("Some players have multiple rows with missing numeric values.")

    # Display the players with multiple missing rows
    offending_values = value_counts[value_counts > 1]
    st.subheader("Players with missing data in multiple rows:")
    st.write(offending_values)

    # Save to CSV for further investigation
    csv_filename = "players_multiple_missing.csv"
    rows_with_missing_values.to_csv(csv_filename, index=False)
    st.success(f"Saved missing values data to {csv_filename}.")


# Identify the count of each player_id
player_counts = st.session_state.main_df['player_id'].value_counts()

# Get player IDs that appear only once
single_occurrence_ids = player_counts[player_counts == 1].index

# Create a mask for rows where player_id occurs only once
single_occurrence_mask = st.session_state.main_df['player_id'].isin(single_occurrence_ids)

# Compute column-wise minimum values for numeric columns
numeric_min = st.session_state.main_df[num_cols].min()

# Fill missing values for players who appear only once using the min of that specific column
st.session_state.main_df.loc[single_occurrence_mask, num_cols] = st.session_state.main_df.loc[single_occurrence_mask, num_cols].fillna(numeric_min)

# Backfill missing values for players who appear multiple times
st.session_state.main_df[num_cols] = st.session_state.main_df[num_cols].fillna(method='bfill')

# Calculate total minutes per team per game and rename that column total_mins
st.session_state.team_game_mins = st.session_state.main_df.groupby(['team', 'game_id'])['mins'].sum().reset_index()
st.session_state.team_game_mins = st.session_state.team_game_mins.rename(columns={'mins': 'total_mins'})

# Merge with main_df
st.session_state.main_df = pd.merge(st.session_state.main_df, st.session_state.team_game_mins, on=['team', 'game_id'], how='left')

# Calculate minutes share and mins_proj
st.session_state.main_df['mins_share'] = st.session_state.main_df['mins'] / st.session_state.main_df['total_mins']
st.session_state.main_df['mins_proj'] = st.session_state.main_df['mins_share'] * 240

#This is the order of features in the model
if st.session_state.positions:
    model_order = ['longName', 'game_id', 'player_id', 'team_id', 'team', 'prim_pos',\
                   'fga', 'ast', 'tptfgm', 'fgm', 'fta', 'tptfga', 'OffReb', 'ftm', 'blk',\
                   'DefReb', 'plusMinus', 'stl', 'pts', 'fouls', 'TOV', 'usage', 'mins_share', 'mins', 'mins_proj', 'salary', 'date'] + st.session_state.positions
    # Reorder the columns in main_df_sorted
    st.session_state.main_df = st.session_state.main_df[model_order]
    #Changing salary to numeric
    st.session_state.main_df['salary'] = pd.to_numeric(st.session_state.main_df['salary'], errors='coerce')

# Filter for today's data only
st.session_state.main_df = st.session_state.main_df[st.session_state.main_df['date'] == today]

# Check for missing values
check_missing_values(st.session_state.main_df)

# Display first and last five rows
st.subheader("First 5 rows of today's data")
st.dataframe(st.session_state.main_df.head())

st.subheader("Last 5 rows of today's data")
st.dataframe(st.session_state.main_df.tail())

# Making a set of the list of all teams playing today
teams_playing = set(list(st.session_state.main_df['team']))
st.write("Teams playing tonight")
st.write(teams_playing)
if st.session_state.teams_in_slate:
    teams_not_in_slate = teams_playing.difference(st.session_state.teams_in_slate)
    st.write("Teams playing tonight that aren't in slate:")
    st.write(sorted(list(teams_not_in_slate)))
    # Filter `main_df` to only include teams in the slate
    st.session_state.main_df = st.session_state.main_df[st.session_state.main_df['team'].isin(st.session_state.teams_in_slate)]

# ✅ Display Main Dataset Before Injury Report
st.subheader("Main Dataset Before Injury Report")
st.dataframe(st.session_state.main_df)  # Display the dataframe before making changes

# 🔹 Button to Fetch Injury Report
if st.button("Fetch Injury Report"):
    slate_ids = st.session_state.main_df["player_id"].tolist()
    st.session_state.injury_df = fetch_injury_report(slate_ids, api_key)

    if not st.session_state.injury_df.empty:
        # Automatically remove 'Out' players from main_df_sorted
        out_players = st.session_state.injury_df[st.session_state.injury_df["status"] == "Out"]["player_id"].tolist()
        st.session_state["removed_players_df"] = st.session_state.main_df[st.session_state.main_df["player_id"].isin(out_players)]
        st.session_state.main_df = st.session_state.main_df[~st.session_state.main_df["player_id"].isin(out_players)]

        # Update session state so changes persist
        # st.session_state["main_df_sorted"] = main_df_sorted
        # st.session_state["injury_df"] = injury_df  # Store injury report in session
        st.success("Injury report fetched. Out players removed.")

# ✅ Display Removed Players
if "removed_players_df" in st.session_state and not st.session_state.removed_players_df.empty:
    st.subheader("Removed Players (Out)")
    st.dataframe(st.session_state["removed_players_df"])
    all_removed_players = list(st.session_state.removed_players_df["player_id"])

# ✅ Display Players with "Day-to-Day" Status
if not st.session_state.injury_df.empty:
    day_to_day_players = st.session_state["injury_df"][st.session_state["injury_df"]["status"] == "Day-To-Day"]

    if not day_to_day_players.empty:
        st.subheader("Players with Day-to-Day Status")
        st.dataframe(day_to_day_players[["name", "status", "injury", "inj_date", "return_date"]])
    else:
        st.info("No players with Day-to-Day status.")

# 🛑 MANUAL PLAYER REMOVAL
st.subheader("Manually Remove Players from Consideration")

# Let user input player names to remove
players_to_remove = st.multiselect(
    "Select players to remove from the slate:",
    options=st.session_state["main_df"]["longName"].unique()
)

# Add a confirmation button to apply removal
if st.button("Confirm Player Removal"):
    if players_to_remove:
        # Identify the removed players
        removed_df = st.session_state["main_df"][st.session_state["main_df"]["longName"].isin(players_to_remove)]

        # Store removed players in session state
        if "removed_players_df" in st.session_state:
            st.session_state["removed_players_df"] = pd.concat([st.session_state["removed_players_df"], removed_df], ignore_index=True)
        else:
            st.session_state["removed_players_df"] = removed_df

        # Remove players from main_df
        st.session_state["main_df"] = st.session_state["main_df"][~st.session_state["main_df"]["longName"].isin(players_to_remove)]

        st.success(f"Removed {len(players_to_remove)} players from the slate.")

    else:
        st.warning("No players selected for removal.")


# # Let user input player names to remove
# players_to_remove = st.multiselect(
#     "Select players to remove from the slate:",
#     options=st.session_state["main_df"]["longName"].unique()
# )

# # If players are selected, remove them from main_df_sorted
# if players_to_remove:
#     removed_df = st.session_state["main_df"][st.session_state["main_df"]["longName"].isin(players_to_remove)]
#     st.session_state["removed_players_df"] = pd.concat([st.session_state["removed_players_df"], removed_df]) 
#     all_removed_players += list(players_to_remove) 

#     # Remove players from dataset
#     st.session_state["main_df"] = st.session_state["main_df"][~st.session_state["main_df"]["player_id"].isin(all_removed_players)]
    
#     st.success(f"Removed {len(players_to_remove)} players from the dataset.")

# 🔹 **Adjust mins_share and mins_proj for Teammates**
if not st.session_state["removed_players_df"].empty:
    team_position_loss = (
        st.session_state["removed_players_df"]
        .groupby(["team", "prim_pos"])[["mins_share", "mins_proj"]]
        .sum()
        .reset_index()
    )
    
    for index, row in team_position_loss.iterrows():
        team, pos, lost_mins_share, lost_mins_proj = row["team"], row["prim_pos"], row["mins_share"], row["mins_proj"]
    
        # Get remaining players at the same position
        mask = (st.session_state["main_df"]["team"] == team) & (st.session_state["main_df"]["prim_pos"] == pos)
        remaining_players = st.session_state["main_df"].loc[mask]
    
        if not remaining_players.empty:
            # Separate scaling for mins_share and mins_proj
            total_existing_share = remaining_players["mins_share"].sum()
            total_existing_proj = remaining_players["mins_proj"].sum()
    
            if total_existing_share > 0:
                # Distribute lost mins_share proportionally
                st.session_state["main_df"].loc[mask, "mins_share"] += (
                    (remaining_players["mins_share"] / total_existing_share) * lost_mins_share
                )
    
            if total_existing_proj > 0:
                # Distribute lost mins_proj proportionally
                st.session_state["main_df"].loc[mask, "mins_proj"] += (
                    (remaining_players["mins_proj"] / total_existing_proj) * lost_mins_proj
                )
    
    st.success("Teammate variables adjusted after manual removal.")


    # # 🔹 **Adjust mins_share and mins_proj for Teammates**
    # team_position_loss = (
    #     st.session_state["removed_players_df"].groupby(["team", "prim_pos"])[["mins_share", "mins_proj"]]
    #     .sum()
    #     .reset_index()
    # )

    # for index, row in team_position_loss.iterrows():
    #     team, pos, lost_mins_share, lost_mins_proj = row["team"], row["prim_pos"], row["mins_share"], row["mins_proj"]

    #     # Get remaining players at the same position
    #     mask = (st.session_state["main_df"]["team"] == team) & (st.session_state["main_df"]["prim_pos"] == pos)
    #     remaining_players = st.session_state["main_df"].loc[mask]

    #     if not remaining_players.empty:
    #         total_existing_share = remaining_players["mins_share"].sum()

    #         if total_existing_share > 0:
    #             # Allocate missing minutes proportionally
    #             st.session_state["main_df"].loc[mask, "mins_share"] += remaining_players["mins_share"] / total_existing_share * lost_mins_share
    #             st.session_state["main_df"].loc[mask, "mins_proj"] += remaining_players["mins_proj"] / total_existing_share * lost_mins_proj

    # st.success("Teammate variables adjusted after manual removal.")

# ✅ Display Updated Player Pool
st.subheader("Updated Player Pool After Removals and Adjustments")
st.dataframe(st.session_state["main_df"])

#EDA
if st.button("Explore Data?"):
    st.subheader("Summary statistics of main_df_sorted")
    st.dataframe(st.session_state.main_df.describe().T.round(3))  
    
    # Generate histograms for each numeric column in a 3-column grid layout
    st.subheader("Histograms of Features")
    
    num_features = st.session_state.main_df.select_dtypes(include=['number']).columns
    num_cols = 3  # Number of columns in the grid
    rows = -(-len(num_features) // num_cols)  # Ceiling division to get the number of rows
    
    for i in range(0, len(num_features), num_cols):
        cols = st.columns(num_cols)  # Create a row with 3 columns
        for j in range(num_cols):
            if i + j < len(num_features):  # Ensure we don't go out of bounds
                col_name = num_features[i + j]
                fig, ax = plt.subplots()
                ax.hist(st.session_state.main_df[col_name].dropna(), bins=30, edgecolor='black')
    
                # Set title and labels with larger font sizes
                ax.set_title(f"{col_name}", fontsize=14, fontweight='bold')  # Increased font size
                ax.set_xlabel(col_name, fontsize=12)  # Bigger x-axis label
                ax.set_ylabel("Frequency", fontsize=12)  # Bigger y-axis label
    
                cols[j].pyplot(fig)  # Display the histogram in the corresponding column

# 🛠️ **Optional Data Editing**
if st.button("Edit Data?"):
    st.write("Make any necessary changes to the dataset before continuing.")

    # Provide an editable version of `main_df`
    edited_df = st.data_editor(st.session_state.main_df, num_rows="dynamic")

    # Save changes when user confirms
    if st.button("Save Changes"):
        st.session_state.main_df = edited_df
        st.success("Edits saved successfully!")

### PREDICTIONS ###

RMSE_FD = 9.683
RMSE_DK = 9.668

pos_cols = []

if st.button("Generate Predictions"):
    if 'site' in st.session_state and st.session_state.site:
        site = st.session_state.site.lower()

        if site == "fanduel":
            model_path = "../best_XGB_FD.pkl"
            scaler_path = "../nba_scaler_fd.pkl"
            rmse = RMSE_FD
            pos_cols = ["PG", "SG", "SF", "PF", "C"]
        else:
            model_path = "../best_XGB_DK.pkl"
            scaler_path = "../nba_scaler_dk.pkl"
            rmse = RMSE_DK
            pos_cols = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

        # Load model and scaler
        with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file) # Rename for model compatibility
        st.session_state.main_df = st.session_state.main_df.rename(columns={'PF': 'PF_pos'})
        X = st.session_state.main_df.rename(columns={'fouls': 'PF'})  # Rename fouls to PF

        # Select features and scale
        expected_feature_order = [
            'fga', 'ast', 'tptfgm', 'fgm', 'fta', 'tptfga', 'OffReb', 'ftm', 'blk',
            'DefReb', 'plusMinus', 'stl', 'pts', 'PF', 'TOV', 'usage', 'mins_share',
            'mins', 'mins_proj'
        ]
        
        X = X[expected_feature_order]
        X_scaled = scaler.transform(X)

        # Predict with the XGBoost model
        predictions = model.predict(X_scaled)

        # Store predictions
        st.session_state.main_df["Pred"] = predictions
        st.session_state.main_df["Floor"] = predictions - rmse
        st.session_state.main_df["Ceiling"] = predictions + rmse

        # Restore PF position column
        st.session_state.main_df = st.session_state.main_df.rename(columns={'PF_pos': 'PF'})

        # Store predictions in session state
        # st.session_state["main_df_sorted"] = main_df_sorted
        # ✅ Calculate "Value" column
        st.session_state["main_df"]["Value"] = (st.session_state["main_df"]["Pred"] / 
                                                       st.session_state["main_df"]["salary"]) * 1000
        st.success("Predictions generated!")


# Display Predictions if available
if "main_df" in st.session_state and "Pred" in st.session_state["main_df"].columns:
    st.subheader("Predictions")

    # Ensure the "Value" column is computed correctly
    # st.session_state["main_df_sorted"]["Value"] = (
    #     st.session_state["main_df_sorted"]['Pred'] / st.session_state["main_df_sorted"]['salary']
    # ) * 1000

    # Merge injury info if available, but only once
    if "injury_df" in st.session_state and not st.session_state["injury_df"].empty:
        st.session_state["main_df"] = st.session_state["main_df"].merge(
            st.session_state["injury_df"][["player_id", "status"]],
            on="player_id",
            how="left"
        )

        # Define columns to display
    display_columns = ["longName", "team", "salary", "Pred", "Floor", "Ceiling", "Value"] + pos_cols
    
    # Ensure 'status' column is included only if it exists in the DataFrame
    if "status" in st.session_state["main_df"].columns:
        display_columns.append("status")
    
    # Display dataframe
    st.dataframe(st.session_state["main_df"][display_columns])

    # Ensure 'status' column is included only if it exists
    # if "status" in st.session_state["main_df_sorted"].columns:
    #     display_columns.append("status")  # Adds injury status to the display

    # # Apply highlighting for "Day-to-Day" players
    # styled_df = st.session_state["main_df_sorted"][display_columns].style.apply(highlight_day_to_day, axis=1)

    # Display dataframe
    #st.dataframe(st.session_state["main_df"][display_columns])

else:
    st.info("Generate predictions first.")



st.write("Session state:")

st.session_state


# if 'my_name' not in st.session_state:
#     st.session_state.my_name = 'Mike'

# st.write(st.session_state)

# st.header("This is a header")

# st.markdown("This is markdown")

# st.subheader("This is a subhead")

# st.code("This is code")

# st.button("Click Me")

# st.radio("Which site do you want to play?", ['FanDuel', 'DraftKings'])

# st.color_picker("Choose a color.")

# st.write("Session state:")

# st.session_state

# number = st.slider('A number', 1, 10, key = 'slider')

# st.write(st.session_state)

# col1, buff, col2 = st.beta_columns([1, 0, 5, 3])

# option_names = ['a', 'b', 'c']

# next = st.button("Next Option")



