import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import streamlit as st
import matplotlib.pyplot as plt
import os
import pulp
import random

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
        df = df[['longName', 'game_id', 'player_id', 'team', 'salary', 'PG', 'SG', 'SF', 'PF', 'C', 'FD_Pred']]
    
    # Check if DK_Pred column is not empty
    elif "DK_Pred" in df.columns and df["DK_Pred"].notna().sum() > 0:
        st.session_state.site = "DK"
        st.session_state.cap = 50000
        df = df[['longName', 'game_id', 'player_id', 'team', 'salary', 'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL', 'DK_Pred']]
    
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



def generate_lineup(df, salary_cap, site, excluded_players=None, locked_players=None, excluded_teams=None):
    if excluded_players is None:
        excluded_players = []
    if locked_players is None:
        locked_players = []
    if excluded_teams is None:
        excluded_teams = []

    prob = pulp.LpProblem("NBA_Lineup_Optimizer", pulp.LpMaximize)
    player_vars = [pulp.LpVariable(f'player_{row.Index}', cat='Binary') for row in df.itertuples()]
    prob += pulp.lpSum(player_var for player_var in player_vars) == 9
    #prob += pulp.lpSum(player_vars[player_id] * predicted_points[player_id] for player_id in df)
    prob += pulp.lpSum(df.salary.iloc[i] * player_vars[i] for i in range(len(df))) <= salary_cap
     # Objective: Maximize total predicted points
    prob += pulp.lpSum([df.FD_Pred.iloc[i] * player_vars[i] for i in range(len(df))])

    def get_position_sum(player_vars, df, position):
        return pulp.lpSum(player_vars[i] * df[position].iloc[i] for i in range(len(df)))
    # PG & SG must sum to at least 4 spots (2 PG, 2 SG), max 5 (to allow flexibility)
    # Force each primary position (PG, SG, SF, PF) to have between 2 and 3 players
    prob += get_position_sum(player_vars, df, 'PG') >= 2
    prob += get_position_sum(player_vars, df, 'PG') <= 3
    
    prob += get_position_sum(player_vars, df, 'SG') >= 2
    prob += get_position_sum(player_vars, df, 'SG') <= 3
    
    prob += get_position_sum(player_vars, df, 'SF') >= 2
    prob += get_position_sum(player_vars, df, 'SF') <= 3
    
    prob += get_position_sum(player_vars, df, 'PF') >= 2
    prob += get_position_sum(player_vars, df, 'PF') <= 3
    
    # Centers must have at least 1, but no more than 2
    prob += get_position_sum(player_vars, df, 'C') == 1


    # ✅ Corrected Position Constraint Example (for PG position)
    # prob += get_position_sum(player_vars, df, 'PG') == 2  # At least 2 PGs
    # prob += get_position_sum(player_vars, df, 'SG') == 2  # At least 2 SGs
    # prob += get_position_sum(player_vars, df, 'SF') == 2  # At least 2 SFs
    # prob += get_position_sum(player_vars, df, 'PF') == 2  # At least 2 PFs
    # prob += get_position_sum(player_vars, df, 'C') == 1  #At least 1 C

    # prob += get_position_sum(player_vars, df, 'PG') + get_position_sum(player_vars, df, 'SG') >= 4  # At least 4 total guards
    # prob += get_position_sum(player_vars, df, 'SF') + get_position_sum(player_vars, df, 'PF') >= 4  # At least 4 total forwards

    # **🚨 Max 4 players per team constraint**
    for team in df['team'].unique():
        prob += pulp.lpSum(player_vars[i] for i in range(len(df)) if df.iloc[i]['team'] == team) <= 4

     # Exclude specific players and entire teams
    for i, row in df.iterrows():
        if row['longName'] in excluded_players or row['team'] in excluded_teams:
            prob += player_vars[i] == 0  # Prevent this player from being selected

    # Lock specific players into the lineup
    for i, row in df.iterrows():
        if row['longName'] in locked_players:
            prob += player_vars[i] == 1  # Force this player to be selected

    prob.solve()

    lineup = []
    total_salary_used = 0
    total_pred_points = 0

    for i in range(len(df)):
        if player_vars[i].value() == 1:
            row = df.iloc[i]
            lineup.append([row['PG'], row['SG'], row['SF'], row['PF'], row['C'], row['longName'], row['team'], row['salary'], row['FD_Pred']])
            total_salary_used += row['salary']
            total_pred_points += row['FD_Pred']

    

                # Return lineup and total info
    return lineup, total_salary_used, total_pred_points

    # lineup = df.sample(n = 9)
    # total_salary = lineup['salary'].sum()
    # total_pred_points = lineup['FD_Pred'].sum()

    return lineup, total_salary, total_pred_points
    
        


if st.button('Generate Lineup'):
    # Call the lineup generation function when the button is clicked
    # Generate the lineup and display it
    site = st.session_state.site
    cap = st.session_state.cap
    lineup, total_salary, total_pred_points = generate_lineup(df, cap, site)
    
    st.markdown(create_subtitle(f"{('FanDuel' if st.session_state.site == 'FD' else 'DraftKings')} Lineup"), unsafe_allow_html=True)
    st.write(f"Total Salary Used: {total_salary}")
    st.write(f"Total Predicted Points: {np.round(total_pred_points, 3)}")
    
    # Create a DataFrame for the lineup with custom column headers
    if st.session_state.site == 'FD':
        lineup_df = pd.DataFrame(lineup, columns = ['PG', 'SG', 'SF', 'PF', 'C', 'Name', 'Team', 'Salary', 'FD_Pred'])
    else: 
        lineup_df = pd.DataFrame(lineup)
    st.table(lineup_df)