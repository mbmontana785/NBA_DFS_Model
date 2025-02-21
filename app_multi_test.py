import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import os
import pulp
import math
import random

def create_subtitle(text, emphasis=True):
    if emphasis:
        return f"<h3 style='color: black; font-weight: bold;'>{text}</h3>"
    else:
        return f"<h3 style='color: black;'>{text}</h3>"

# Define CSS for each site, FanDuel or DraftKings
fd_css = """
    <style>
        .stApp { background-color: #f8f9fa !important; }
        h1 { color: #1493FF !important; }
        h2, h3, h4, h5, h6 { color: #2CB459 !important; }
        .stButton>button { background-color: #0044cc !important; color: white !important; border-radius: 5px; }
        .stSelectbox div { background-color: white !important; color: #003366 !important; }
    </style>
"""

dk_css = """
    <style>
        .stApp { background-color: #0E1117 !important; color: #CCCCCC !important; }
        
        /* Adjust header colors */
        h1 { color: #9AC434 !important; }  /* Darker green */
        h2, h3, h4, h5, h6 { color: #F46C22 !important; }  /* Dark orange */

        /* Improve table readability */
        table {
            background-color: #161B22 !important;
            color: #FFFFFF !important;  /* White text for contrast */
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            border: 1px solid #444444 !important;
            padding: 8px;
            text-align: left;
            color: #FFFFFF !important;  /* Ensure text is visible */
        }

        th {
            background-color: #1F2937 !important;
            color: #9AC434 !important;  /* Dark green headers */
        }

        /* Buttons */
        .stButton>button { background-color: #006400 !important; color: white !important; border-radius: 5px; }

        /* Dropdowns & Selectboxes */
        .stSelectbox div { background-color: black !important; color: #00ff00 !important; }

        /* General text */
        p, span, label { color: #CCCCCC !important; }  /* Lighter gray text for readability */
    </style>
"""
        
       
# Display today's date
today = datetime.now().strftime("%Y%m%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
last_day = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")
formatted_date = datetime.now().strftime("%B %#d, %Y")

# Streamlit App
st.title(f"NBA DFS Lineup Generator for {formatted_date}")

# Initialize session state for main_df if not already set
if "main_df" not in st.session_state:
    st.session_state.main_df = pd.DataFrame()

if "site" not in st.session_state:
    st.session_state.site = None

if "cap" not in st.session_state:
    st.session_state.cap = None

if "all_lineups_df" not in st.session_state:
    st.session_state.all_lineups_df = pd.DataFrame()

if "exclude_list" not in st.session_state:
    st.session_state.exclude_list = []

if "lock_list" not in st.session_state:
    st.session_state.lock_list = []

if "exclude_teams" not in st.session_state:
    st.session_state.exclude_teams = []

# exclude_list = []
# lock_list = []
# exclude_teams = []


# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Read the CSV into session state when uploaded
if uploaded_file is not None:
    st.session_state.main_df = pd.read_csv(uploaded_file)
    if 'FD_Pred' in st.session_state.main_df.columns:
        st.success("You're playing FanDuel!")
    else:
        st.success("You're playing DraftKings!")

    # Fill missing values in relevant numerical columns
    for col in ["FD_Pred", "DK_Pred", "FD_Value", "DK_Value", "salary"]:
        if col in st.session_state.main_df.columns:
            st.session_state.main_df[col].fillna(0, inplace=True)  

    # Ensure no missing values in 'longName' or 'team' (critical for constraints)
    st.session_state.main_df.dropna(subset=["longName", "player_id", "team"], inplace=True)

    # Check for any remaining NaNs
    if st.session_state.main_df.isna().sum().sum() > 0:
        st.warning("Some missing values remain in the dataset. Check the source data.")

# Display the first 5 rows of the dataframe if loaded
if not st.session_state.main_df.empty:
    st.subheader("Preview of Loaded Data")
    st.dataframe(st.session_state.main_df.head())

if "main_df" in st.session_state:
    df = st.session_state.main_df  # Alias for easier use
    
    
    # Check if FD_Pred column is not empty
    if "FD_Pred" in df.columns and df["FD_Pred"].notna().sum() > 0:
        st.session_state.site = "FD"
        st.session_state.cap = 60000
        df = df[['longName', 'game_id', 'player_id', 'team', 'salary', 'PG', 'SG', 'SF', 'PF', 'C', 'FD_Pred', 'FD_Value']]
        df["player_id"] = df["player_id"].astype(str)
    
    # Check if DK_Pred column is not empty
    elif "DK_Pred" in df.columns and df["DK_Pred"].notna().sum() > 0:
        st.session_state.site = "DK"
        st.session_state.cap = 50000
        df = df[['longName', 'game_id', 'player_id', 'team', 'salary', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'DK_Pred', 'DK_Value']]
        df["player_id"] = df["player_id"].astype(str)
    
    else:
        st.session_state.site = None  # No valid predictions found
    
# Apply the appropriate theme **AFTER site detection**
if st.session_state.site == "FD":
    st.markdown(fd_css, unsafe_allow_html=True)
elif st.session_state.site == "DK":
    st.markdown(dk_css, unsafe_allow_html=True)

if not st.session_state.main_df.empty:
    # Filter by team
    team_filter = st.multiselect("Filter by team:", options=df['team'].unique(), default=df['team'].unique())
    
    # Apply the filters to the DataFrame
    filtered_df = df[df['team'].isin(team_filter)]
    
    st.dataframe(filtered_df)


if not st.session_state.main_df.empty:
    # Select players to lock into the lineup
    st.session_state.lock_list = st.multiselect("Select players to lock into the lineup:", df['longName'].unique())
    
    # Select players to exclude from the lineup
    st.session_state.exclude_list = st.multiselect("Select players to exclude from the lineup:", df['longName'].unique())
    
    # Select teams to exclude from the lineup
    st.session_state.exclude_teams = st.multiselect("Select teams to exclude from the lineup:", df['team'].unique())


def generate_lineup(df, salary_cap, site, locked_players=None, excluded_teams=None, limit=None):
    # if excluded_players is None:
    #     excluded_players = []
    if locked_players is None:
        locked_players = []
    if excluded_teams is None:
        excluded_teams = []
    
    prob = pulp.LpProblem("NBA_Lineup_Optimizer", pulp.LpMaximize)
    player_vars = [pulp.LpVariable(f'player_{row.Index}', cat='Binary') for row in df.itertuples()]
    if site == 'FD':
        prob += pulp.lpSum(player_var for player_var in player_vars) == 9
    else:
        prob += pulp.lpSum(player_var for player_var in player_vars) == 8
        
    prob += pulp.lpSum(df.salary.iloc[i] * player_vars[i] for i in range(len(df))) <= salary_cap
    
     # Objective: Maximize total predicted points
    if site == 'FD':
        prob += pulp.lpSum([df.FD_Pred.iloc[i] * player_vars[i] for i in range(len(df))])
    else:
        prob += pulp.lpSum([df.DK_Pred.iloc[i] * player_vars[i] for i in range(len(df))])
    
    def get_position_sum(player_vars, df, position):
        return pulp.lpSum(player_vars[i] * df[position].iloc[i] for i in range(len(df)))

    # **New Constraint: Ensure lineup is different by limiting predicted points**
    if point_limit is not None:
        if site == 'FD':
            prob += pulp.lpSum(df.FD_Pred.iloc[i] * player_vars[i] for i in range(len(df))) <= point_limit - .01
        else:
            prob += pulp.lpSum(df.DK_Pred.iloc[i] * player_vars[i] for i in range(len(df))) <= point_limit - .01
   
    if site == 'FD':
        prob += get_position_sum(player_vars, df, 'PG') >= 2
        prob += get_position_sum(player_vars, df, 'SG') >= 4
        prob += get_position_sum(player_vars, df, 'SF') >= 4 
        prob += get_position_sum(player_vars, df, 'PF') >= 4
        prob += get_position_sum(player_vars, df, 'C') >= 1
        
          # **🚨 Max 4 players per team constraint**
        for team in df['team'].unique():
            prob += pulp.lpSum(player_vars[i] for i in range(len(df)) if df.iloc[i]['team'] == team) <= 4    
    else:
        prob += get_position_sum(player_vars, df, 'PG') >= 1
        prob += get_position_sum(player_vars, df, 'SG') >= 1
        prob += get_position_sum(player_vars, df, 'G') >= 3
        prob += get_position_sum(player_vars, df, 'SF') >= 1
        prob += get_position_sum(player_vars, df, 'PF') >= 1
        prob += get_position_sum(player_vars, df, 'F') >= 3
        prob += get_position_sum(player_vars, df, 'C') >= 1 
        prob += get_position_sum(player_vars, df, 'C') <= 2
    
    
     # Exclude specific players and entire teams
    for i, row in df.iterrows():
        if row['team'] in excluded_teams:
            prob += player_vars[i] == 0  # Prevent this player from being selected
    
    # Lock specific players into the lineup
    for i, row in df.iterrows():
        if row['longName'] in locked_players:
            prob += player_vars[i] == 1  # Force this player to be selected
    
    # Solve the problem
    prob.solve()
    
    # Debugging information in Streamlit
    # st.subheader("Debugging Information")
    # st.write("### Player Variables (Selected Players)")
    # for i in range(len(df)):
    #     if player_vars[i].varValue is not None and player_vars[i].varValue > 0:  
    #         st.write(f"{player_vars[i].name}: {player_vars[i].varValue}")
    
    # st.write("### Objective Function (Maximize Total Predicted Points)")
    # st.write(prob.objective)
    
    # Initialize lineup lists
    lineup = []
    total_salary_used = 0
    total_pred_points = 0
    
    if site == 'FD':
        for i in range(len(df)):
            if player_vars[i].varValue == 1:  # Check the selected players
                row = df.iloc[i]
                lineup.append([
                    row['PG'], row['SG'], row['SF'], row['PF'], row['C'], 
                    row['player_id'], row['longName'], row['team'], row['salary'], row['FD_Pred'], row['FD_Value']
                ])
                total_salary_used += row['salary']
                total_pred_points += row['FD_Pred']
    else:
        for i in range(len(df)):
            if player_vars[i].varValue == 1:  # Check the selected players
                row = df.iloc[i]
                lineup.append([
                    row['PG'], row['SG'], row['SF'], row['PF'], row['C'], row['G'], row['F'], row['UTIL'], 
                    row['player_id'], row['longName'], row['team'], row['salary'], row['DK_Pred'], row['DK_Value']
                ])
                total_salary_used += row['salary']
                total_pred_points += row['DK_Pred']
    
    # Return lineup and total info
    return lineup, total_salary_used, total_pred_points
    

def fill_positions(site, lineup_df):
    """
    Assigns each player a single position based on DFS site rules while allowing backtracking 
    to correct mistakes when needed.
    
    Args:
        site (str): 'FD' for FanDuel, 'DK' for DraftKings.
        lineup_df (pd.DataFrame): The DataFrame containing the lineup.

    Returns:
        lineup_df (pd.DataFrame): Updated DataFrame with assigned positions in 'LU_Pos'.
    """

    if site == 'FD':
        position_order = ['PG', 'PG', 'SG', 'SG', 'SF', 'SF', 'PF', 'PF', 'C']
    else:
        position_order = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']

    lineup_df['LU_Pos'] = None  # Initialize position assignments

    def is_valid_assignment(assignment):
        """Checks if the current assignment of positions satisfies DFS rules."""
        position_counts = assignment.value_counts().to_dict()
        required_positions = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1} if site == "FD" else \
                             {'PG': 1, 'SG': 1, 'G': 1, 'SF': 1, 'PF': 1, 'F': 1, 'C': 1, 'UTIL': 1}
        
        return all(position_counts.get(pos, 0) >= required_positions[pos] for pos in required_positions)

    def backtrack(index, assigned_positions):
        """
        Tries to assign positions recursively and backtracks if conflicts arise.
        
        Args:
            index (int): Current position in position_order we're trying to assign.
            assigned_positions (dict): Keeps track of assigned positions for players.

        Returns:
            bool: True if a valid assignment was found, False otherwise.
        """
        if index == len(position_order):  # If all positions assigned, check validity
            return is_valid_assignment(pd.Series(assigned_positions))

        pos = position_order[index]
        available_players = lineup_df[
            (lineup_df[pos] == 1) & (lineup_df['LU_Pos'].isna())
        ]

        for idx in available_players.index:
            assigned_positions[idx] = pos  # Assign the position
            lineup_df.at[idx, 'LU_Pos'] = pos

            if backtrack(index + 1, assigned_positions):  # Recur to assign next position
                return True  # If a valid lineup is found, return True

            # Undo assignment (backtracking step)
            assigned_positions.pop(idx)
            lineup_df.at[idx, 'LU_Pos'] = None

        return False  # If no valid assignments, return False

    # Start backtracking from the first position
    success = backtrack(0, {})

    if not success:
        print("Warning: Could not find a valid position assignment!")

    return lineup_df



            
num_lineups = st.number_input("How many lineups would you like to generate?", min_value=1, max_value=150, step=1)
if num_lineups > 1:
     # Get the maximum exposure percentage as an integer from user input
        max_exposure_percentage = st.number_input("What's the maximum exposure percentage?", min_value=1, max_value=100, value=50, step=1)

        # Compute max exposure by rounding down the number of times a player can appear
        max_exposure = math.floor((max_exposure_percentage / 100) * num_lineups) 
    
if st.button('Generate Lineup'):
    # Call the lineup generation function when the button is clicked
    # Generate the lineup and display it
    site = st.session_state.site
    cap = st.session_state.cap
    point_limit = None
    if num_lineups == 1:
        #Run everything normally if we generate just one lineup
        df = df[~df["longName"].isin(st.session_state.exclude_list)]
        lineup, total_salary, total_pred_points = generate_lineup(df, cap, site, st.session_state.lock_list, st.session_state.exclude_teams, point_limit)
        
        st.markdown(create_subtitle(f"{('FanDuel' if st.session_state.site == 'FD' else 'DraftKings')} Lineup"), unsafe_allow_html=True)
        st.write(f"Total Salary Used: {total_salary}")
        st.write(f"Total Predicted Points: {np.round(total_pred_points, 3)}")
        
        # Create a DataFrame for the lineup with custom column headers
        if st.session_state.site == 'FD':
            lineup_df = pd.DataFrame(lineup, columns = ['PG', 'SG', 'SF', 'PF', 'C', 'player_id', 'Name', 'Team', 'Salary', 'FD_Pred', 'FD_Value'])
        else: 
            lineup_df = pd.DataFrame(lineup, columns = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'player_id', 'Name', 'Team', 'Salary', 'DK_Pred', 'DK_Value'])
        st.table(lineup_df)
    
        # Save lineup to CSV
        csv = lineup_df.to_csv(index=False).encode('utf-8')
            
        # Provide a download button for the lineup CSV
        st.download_button(
            label="Download Lineup CSV",
            data=csv,
            file_name=f"{st.session_state.site}_lineup_{today}.csv",
            mime="text/csv",
        )
    else:
        df = df[~df["longName"].isin(st.session_state.exclude_list)]
        for i in range(num_lineups):
            if not st.session_state.all_lineups_df.empty:
                exposure_counts = st.session_state.all_lineups_df["player_id"].value_counts()
                overexposed_players = exposure_counts[exposure_counts >= max_exposure].index
                df = df[~df["player_id"].isin(overexposed_players)]  # Remove overexposed players
            if i > 0:
                point_limit = total_pred_points
            lineup, total_salary, total_pred_points = generate_lineup(df, cap, site, st.session_state.lock_list, st.session_state.exclude_teams, point_limit)
            if st.session_state.site == 'FD':
                lineup_df = pd.DataFrame(lineup, columns = ['PG', 'SG', 'SF', 'PF', 'C', 'player_id', 'Name', 'Team', 'Salary', 'FD_Pred', 'FD_Value'])
                lineup_df['lineup_id'] = i
            else: 
                lineup_df = pd.DataFrame(lineup, columns = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'player_id', 'Name', 'Team', 'Salary', 'DK_Pred', 'DK_Value'])
                lineup_df['lineup_id'] = i
            lineup_df = fill_positions(site, lineup_df)
            st.session_state.all_lineups_df = pd.concat([lineup_df, st.session_state.all_lineups_df]) 
        
        # Count invalid lineups (those missing required positions)
        invalid_lineups = 0
        
        for lineup_id in st.session_state.all_lineups_df["lineup_id"].unique():
            lineup_subset = st.session_state.all_lineups_df[st.session_state.all_lineups_df["lineup_id"] == lineup_id]
            
            required_positions = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1} if site == "FD" else \
                                 {'PG': 1, 'SG': 1, 'G': 1, 'SF': 1, 'PF': 1, 'F': 1, 'C': 1, 'UTIL': 1}
            
            position_counts = lineup_subset["LU_Pos"].value_counts().to_dict()
            
            for pos, min_count in required_positions.items():
                if position_counts.get(pos, 0) < min_count:
                    invalid_lineups += 1
                    break  # No need to check further for this lineup
        
        st.write(f"Valid lineups detected: {num_lineups - invalid_lineups} out of {num_lineups}")

        # Save lineup to CSV
        csv = st.session_state.all_lineups_df.to_csv(index=False).encode('utf-8')
            
        # Provide a download button for the lineup CSV
        st.download_button(
            label="Download Lineup CSVs",
            data=csv,
            file_name=f"{st.session_state.site}_lineups_{today}.csv",
            mime="text/csv",
        )

view_id = st.number_input("Enter a number (starting with 0) to view that lineup.", min_value=0, max_value=149, step=1, key="view_id")

if not st.session_state.all_lineups_df.empty:
    view_df = st.session_state.all_lineups_df[st.session_state.all_lineups_df['lineup_id'] == view_id]

    st.markdown(create_subtitle(f"{'FanDuel' if st.session_state.site == 'FD' else 'DraftKings'} Lineup"), unsafe_allow_html=True)
    st.write(f"Total Salary Used: {view_df['Salary'].sum()}")

    if st.session_state.site == 'FD':
        view_df = pd.DataFrame(view_df, columns=['PG', 'SG', 'SF', 'PF', 'C', 'player_id', 'Name', 'Team', 'Salary', 'FD_Pred', 'FD_Value', 'LU_Pos'])
        st.write(f"Total Predicted Points: {np.round(view_df['FD_Pred'].sum(), 3)}")
    else: 
        view_df = pd.DataFrame(view_df, columns=['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'player_id', 'Name', 'Team', 'Salary', 'DK_Pred', 'DK_Value', 'LU_Pos'])
        st.write(f"Total Predicted Points: {np.round(view_df['DK_Pred'].sum(), 3)}")

    st.table(view_df)


# Allow user to check lineup exposure only if data is available
if not st.session_state.all_lineups_df.empty:
    check_exposure = st.multiselect(
        "Select players to check lineup exposure:", 
        st.session_state.all_lineups_df["Name"].unique()
    )

    if check_exposure:  # Proceed only if players are selected
        # Count the number of lineups each selected player appears in
        lineup_counts = (
            st.session_state.all_lineups_df[st.session_state.all_lineups_df["Name"].isin(check_exposure)]
            .groupby("Name")["lineup_id"]
            .nunique()
            .reset_index()
            .rename(columns={"lineup_id": "Lineup Count"})
        )

        # Display results
        st.subheader("Player Lineup Exposure")
        st.table(lineup_counts)
     


