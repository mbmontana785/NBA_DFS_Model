import pandas as pd
import sqlite3
import requests
import streamlit as st
from datetime import datetime, timedelta

def set_yesterday_only(yesterday, last_day):
    conn = sqlite3.connect('nba_dfs_model.db')
    cursor = conn.cursor()
    try:
        query = "SELECT DISTINCT game_id FROM game_stats ORDER BY game_id DESC LIMIT 1"
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            return False  # Database is empty; fetch all data.

        for row in results:
            if row[0][:8] == yesterday:
                return None  # Already up-to-date.
            elif row[0][:8] == last_day:
                return True  # Only yesterday's data is missing.
            else:
                return False  # Data older than yesterday is also missing.
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None
    finally:
        conn.close()

def fetch_game_ids(api_key, date=None, yesterday=None, last_day=None, today=None, yesterday_only=None):
    """
    Fetch game IDs for a specific date, a date range, or based on yesterday_only logic.
    """
    game_ids = []
    no_game_dates = []
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }

    # Fetch game IDs for a single specific date (e.g., today)
    if date:
        url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate?gameDate={date}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

            if 'body' in result and result['body']:
                for game in result['body']:
                    game_ids.append(game['gameID'])
            else:
                no_game_dates.append(date)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {date}: {e}")
        except KeyError:
            st.error(f"Unexpected response format for {date}.")
        return game_ids, no_game_dates

    # Existing logic for yesterday_only or multi-day fetch
    if yesterday_only:
        # Fetch game IDs for yesterday
        url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate?gameDate={yesterday}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

            if 'body' in result and result['body']:
                for game in result['body']:
                    game_ids.append(game['gameID'])
            else:
                no_game_dates.append(yesterday)
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {yesterday}: {e}")
        except KeyError:
            st.error(f"Unexpected response format for {yesterday}.")
    elif yesterday_only == False:
        # Fetch game IDs for multiple days
        start_date = datetime.strptime(last_day, '%Y%m%d')
        end_date = datetime.strptime(today, '%Y%m%d')
        delta = end_date - start_date

        for i in range(1, delta.days):
            current_date = (start_date + timedelta(days=i)).strftime('%Y%m%d')
            url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForDate?gameDate={current_date}"
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                result = response.json()

                if 'body' in result and result['body']:
                    for game in result['body']:
                        game_ids.append(game['gameID'])
                else:
                    no_game_dates.append(current_date)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data for {current_date}: {e}")
            except KeyError:
                st.error(f"Unexpected response format for {current_date}.")
    return game_ids, no_game_dates


def get_row_count():
    try:
        with sqlite3.connect("nba_dfs_model.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM game_stats")
            count = cursor.fetchone()[0]
        return count
    except sqlite3.Error as e:
        st.error(f"Error accessing database: {e}")
        return 0

def update_game_stats(api_key, game_ids):
    rows_added = 0
    games_processed = 0
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }

    with sqlite3.connect("nba_dfs_model.db") as conn:
        cursor = conn.cursor()

        for game_id in game_ids:
            try:
                url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBABoxScore?gameID={game_id}"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                result = response.json()

                if 'body' not in result or 'playerStats' not in result['body']:
                    st.warning(f"No player stats found for game ID {game_id}. Skipping...")
                    continue

                current_dict = result['body']['playerStats']

                for player_id, stats in current_dict.items():
                    try:
                        cursor.execute('''
                        INSERT OR REPLACE INTO game_stats (
                            longName, game_id, player_id, team_id, team, teamAbv, fga, ast, tptfgm, fgm, fta, tptfga,
                            OffReb, ftm, blk, DefReb, plusMinus, stl, pts, PF, TOV, usage, mins
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        ''', (
                            stats.get('longName', ''), stats['gameID'], stats['playerID'], stats.get('teamID', ''),
                            stats.get('team', ''), stats.get('teamAbv', ''), stats.get('fga', 0), stats.get('ast', 0),
                            stats.get('tptfgm', 0), stats.get('fgm', 0), stats.get('fta', 0), stats.get('tptfga', 0),
                            stats.get('OffReb', 0), stats.get('ftm', 0), stats.get('blk', 0), stats.get('DefReb', 0),
                            stats.get('plusMinus', ''), stats.get('stl', 0), stats.get('pts', 0), stats.get('PF', 0),
                            stats.get('TOV', 0), stats.get('usage', 0.0), stats.get('mins', 0)
                        ))
                        rows_added += 1
                    except KeyError as e:
                        st.error(f"KeyError: {e} for player {player_id} in game {game_id}")
                    except sqlite3.Error as e:
                        st.error(f"SQL Error: {e}")

                conn.commit()
                games_processed += 1
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed for game ID {game_id}: {e}")
            except sqlite3.Error as e:
                st.error(f"Database error: {e}")

    st.success(f"Database updated with {rows_added} rows added from {games_processed} games.")


# Function to check for missing values and save the entire DataFrame only if missing values exist
def check_missing_values(df):
    """
    Check for missing values in a DataFrame.
    If missing values are found, display a summary, save the entire DataFrame as a CSV,
    and provide an option to replace missing values with 0 if they are in numeric columns.
    """
    missing_summary = df.isna().sum()  # Count missing values per column
    columns_with_missing = missing_summary[missing_summary > 0]  # Filter columns with missing values

    if not columns_with_missing.empty:
        st.warning("Missing values detected!")

        # Display summary of missing values
        st.subheader("Missing Values Summary")
        st.write(columns_with_missing)

        # Save the entire DataFrame if missing values exist
        missing_values_file = "full_dataframe_with_missing_values.csv"
        df.to_csv(missing_values_file, index=False)
        st.success(f"Full DataFrame saved as {missing_values_file} for further inspection.")

        # Get the columns that contain missing values
        missing_cols = columns_with_missing.index.tolist()

        # Check if all missing values are in numeric columns
        if df[missing_cols].dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            st.info("All missing values are in numeric columns.")

            # Button to replace NaN with 0
            if st.button("Replace Missing Values with 0"):
                df.fillna(0, inplace=True)
                st.success("Missing values replaced with 0!")

                # Refresh missing value check after replacement
                return check_missing_values(df)
        else:
            st.info("Some missing values are in non-numeric columns. Handle manually.")

    # If no missing values, do nothing and let the program continue silently
    return



def fetch_player_salaries(api_key, today, site, positions):
    """
    Fetch player salaries from the API for the given site and date.
    """
    url = f"https://tank01-fantasy-stats.p.rapidapi.com/getNBADFS?date={today}"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }

    players = []  # List to store player data
    heads = ['pos', 'salary', 'longName', 'player_id', 'team_id', 'team']

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
        result = response.json()

        if 'body' in result and site in result['body']:
            for player in result['body'][site]:
                players.append([
                    player['allValidPositions'], player['salary'], player['longName'], 
                    player['playerID'], player['teamID'], player['team']
                ])
        else:
            st.warning("No player data found for the given date.")
            return None  # Return None if no data is found
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching player data: {e}")
        return None
    except KeyError:
        st.error(f"Unexpected response format: {result}")
        return None

    # Create DataFrame
    today_df = pd.DataFrame(players, columns=heads)

    # Add empty position columns
    for position in positions:
        today_df[position] = 0

    return today_df

def populate_position_columns(today_df, positions, site):
    """
    Populate binary values for position columns and handle special positions for DraftKings.
    """
    # Fill binary values for the position columns
    for position in positions:
        today_df[position] = today_df['pos'].apply(lambda x: position in x)

    # Handle special positions for DraftKings
    if site == 'draftkings':
        today_df['G'] = np.where(today_df['PG'] + today_df['SG'] > 0, True, False)
        today_df['F'] = np.where(today_df['SF'] + today_df['PF'] > 0, True, False)
        today_df['UTIL'] = True

    return today_df

def update_main_df(main_df_sorted, today_df):
    """
    Concatenates today's data with the existing dataset, sorts by player_id, date, and game_id,
    and returns the updated DataFrame.
    """
    updated_df = pd.concat([main_df_sorted, today_df], ignore_index=True)
    updated_df = updated_df.sort_values(['player_id', 'date', 'game_id']).reset_index(drop=True)
    return updated_df

def calculate_rolling_averages(df, num_cols):
    """
    Computes the rolling mean of numerical columns over the last 15 games for each player.
    The shift(1) ensures that the current game is not included in the calculation.
    """
    df[num_cols] = (
        df.groupby('player_id')[num_cols]
        .apply(lambda x: x.shift(1).rolling(window=15, min_periods=1).mean())
        .reset_index(drop=True)
    )
    return df
