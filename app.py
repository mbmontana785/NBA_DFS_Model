import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import os
import pulp

def create_subtitle(text, emphasis=True):
    if emphasis:
        return f"<h3 style='color: black; font-weight: bold;'>{text}</h3>"
    else:
        return f"<h3 style='color: black;'>{text}</h3>"


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

exclude_list = []
lock_list = []
exclude_teams = []


# File uploader widget in Streamlit
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Read the CSV into session state when uploaded
if uploaded_file is not None:
    st.session_state.main_df = pd.read_csv(uploaded_file)
    st.success("CSV file loaded successfully!")

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
    
    # Check if DK_Pred column is not empty
    elif "DK_Pred" in df.columns and df["DK_Pred"].notna().sum() > 0:
        st.session_state.site = "DK"
        st.session_state.cap = 50000
    
    else:
        st.session_state.site = None  # No valid predictions found
    
    #st.write(f"**You're playing ** {st.session_state.site}")  # Display detected site

if not st.session_state.main_df.empty:
    # Filter by team
    team_filter = st.multiselect("Filter by team:", options=df['team'].unique(), default=df['team'].unique())
    
    # Apply the filters to the DataFrame
    filtered_df = df[df['team'].isin(team_filter)]
    
    st.dataframe(filtered_df)


if not st.session_state.main_df.empty:
    # Select players to lock into the lineup
    lock_list = st.multiselect("Select players to lock into the lineup:", df['longName'].unique())
    
    # Select players to exclude from the lineup
    exclude_list = st.multiselect("Select players to exclude from the lineup:", df['longName'].unique())
    
    # Select teams to exclude from the lineup
    exclude_teams = st.multiselect("Select teams to exclude from the lineup:", df['team'].unique())



def generate_lineup(df, salary_cap, site, excluded_players=exclude_list,locked_players=lock_list, excluded_teams=exclude_teams):
    if excluded_players is None:
        excluded_players = []
    if locked_players is None:
        locked_players = []
    if excluded_teams is None:
        excluded_teams = []

    # Create the pulp problem
    prob = pulp.LpProblem('NBA_DFS_Lineup', pulp.LpMaximize)

    # Create variables for each player indicating whether they are included in the lineup
    player_vars = [pulp.LpVariable(f'player_{row.Index}', cat='Binary') for row in df.itertuples()]

    # Total assigned players constraint
    prob += pulp.lpSum(player_var for player_var in player_vars) == 9

    #https://stackoverflow.com/questions/66326293/python-pulp-positional-constraints-for-a-lineup-generator

    #salaries = dict(zip(list(df['longName']), list(df['salary'])))
    
    # # Create variables for each player indicating whether they are included in the lineup
    # player_vars = LpVariable.dicts("assignment",
    # [(i, pos) for i in range(len(df)) for pos in ['PG', 'SG', 'SF', 'PF', 'C'] if df.iloc[i][pos]],
    # cat="Binary")

    #player_vars = {i: pulp.LpVariable(f"player_{i}", cat="Binary") for i in range(len(df))}
    player_vars = [pulp.LpVariable(f'player_{i}', cat='Binary') for i in range(len(df))]

    # Total assigned players constraint
    if site == 'FD':
        prob += pulp.lpSum(player_var for player_var in player_vars) == 9    
    else:
        prob += pulp.lpSum(player_var for player_var in player_vars) == 8

    # Total salary constraint using the provided salary cap
    #prob += pulp.lpSum([df.at[i, "salary"] * player_vars[i] for i in range(len(df))]) <= salary_cap


    # Create a helper function to return the number of players assigned each position
    # def get_position_sum(player_vars, df, position):
    #     """
    #     Returns the sum of players who are eligible for a given position.
    #     This accounts for multi-position eligibility in NBA DFS lineups.
    #     """
    #     return pulp.lpSum(player_vars[i] for i in range(len(df)) if df.iloc[i][position])

    # for i in range(len(df)):
    #     prob += pulp.lpSum(player_vars[(i, pos)] for pos in ['PG', 'SG', 'SF', 'PF', 'C'] if df.iloc[i][pos]) <= 1



    # if site == "FD":
    #     prob += pulp.lpSum(player_vars[(i, "PG")] for i in range(len(df)) if df.iloc[i]["PG"]) == 2
    #     prob += pulp.lpSum(player_vars[(i, "SG")] for i in range(len(df)) if df.iloc[i]["SG"]) == 2
    #     prob += pulp.lpSum(player_vars[(i, "SF")] for i in range(len(df)) if df.iloc[i]["SF"]) == 2
    #     prob += pulp.lpSum(player_vars[(i, "PF")] for i in range(len(df)) if df.iloc[i]["PF"]) == 2
    #     prob += pulp.lpSum(player_vars[(i, "C")] for i in range(len(df)) if df.iloc[i]["C"]) == 1
    # else:  # DK
    #     prob += pulp.lpSum(player_vars[(i, "PG")] for i in range(len(df)) if df.iloc[i]["PG"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "SG")] for i in range(len(df)) if df.iloc[i]["SG"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "SF")] for i in range(len(df)) if df.iloc[i]["SF"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "PF")] for i in range(len(df)) if df.iloc[i]["PF"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "C")] for i in range(len(df)) if df.iloc[i]["C"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "G")] for i in range(len(df)) if df.iloc[i]["G"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "F")] for i in range(len(df)) if df.iloc[i]["F"]) >= 1
    #     prob += pulp.lpSum(player_vars[(i, "UTIL")] for i in range(len(df)) if df.iloc[i]["UTIL"]) >= 1



    # # Objective: Maximize total predicted points
    # if site == 'FD':
    #     prob += pulp.lpSum([df.at[i, "FD_pred"] * player_vars[i] for i in range(len(df))])
    # else:
    #     prob += pulp.lpSum([df.at[i, "DK_pred"] * player_vars[i] for i in range(len(df))])

        
    # # Exclude specific players and entire teams
    # for i, row in df.iterrows():
    #     if row['longName'] in excluded_players or row['team'] in excluded_teams:
    #         prob += player_vars[i] == 0  # Prevent this player from being selected

    # # Lock specific players into the lineup
    # for i, row in df.iterrows():
    #     if row['longName'] in locked_players:
    #         prob += player_vars[i] == 1  # Force this player to be selected

    # Solve the problem
    prob.solve()

    if pulp.LpStatus[prob.status] != "Optimal":
        st.error("No valid lineup found. Adjust constraints and try again.")
        return None, 0, 0  # Instead of calling function again

    # if status != pulp.LpStatusOptimal:
    #     st.error("No valid lineup found. Try relaxing constraints.")
    #     return None, 0, 0

    # Gather lineup details
    lineup = []
    total_salary_used = 0
    total_pred_points = 0

    if site == 'FD':
        for i in range(len(df)):
            if player_vars[i].value() == 1:
                row = df.iloc[i]
                lineup.append([row['PG'], row['SG'], row['SF'], row['PF'], row['C'],\
                row['longName'], row['team'], row['salary'], row['FD_Pred']])
                total_salary_used += row['salary']
                total_pred_points += row['FD_Pred']
    else:
        for i in range(len(df)):
            if player_vars[i].value() == 1:
                row = df.iloc[i]
                lineup.append([row['PG'], row['SG'], row['SF'], row['PF'], row['C'],\
                row['G'], row['F'], row['UTIL'], row['longName'], row['team'], row['salary'], row['DK_Pred']])
                total_salary_used += row['salary']
                total_pred_points += row['DK_Pred']

    # Return lineup and total info
    return lineup, total_salary_used, total_pred_points

if st.button('Generate Lineup'):
    # Call the lineup generation function when the button is clicked
    # Generate the lineup and display it
    site = st.session_state.site
    cap = st.session_state.cap
    lineup, total_salary, total_pred_points = generate_lineup(df, site, cap)
    
    st.markdown(create_subtitle(f"{('FanDuel' if st.session_state.site == 'FD' else 'DraftKings')} Lineup"), unsafe_allow_html=True)
    st.write(f"Total Salary Used: {total_salary}")
    st.write(f"Total Predicted Points: {np.round(total_pred_points, 3)}")
    
    # Create a DataFrame for the lineup with custom column headers
    if st.session_state.site == 'FD':
        lineup_df = pd.DataFrame(lineup, columns=["PG", "SG", "SF", "PF", "C", "Player", "Team", "Salary", "Predicted Points"])
    else: lineup_df = pd.DataFrame(lineup, columns=["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "Player", "Team", "Salary", "Predicted Points"])
    st.table(lineup_df)