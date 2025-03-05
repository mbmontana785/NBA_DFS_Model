# FanDuel and DraftKings NBA DFS Model and Lineup Optimizer

**Optimizer app in Streamlit with Hex color codes adjusted depending on whether we're playing FanDuel or DraftKings:**

<p align="center">
  <img src="https://github.com/user-attachments/assets/b46f34b3-30bc-4d9a-ad35-201e3b2a9812" width="45%">
  <img src="https://github.com/user-attachments/assets/58848afe-fa2b-44d0-b032-f37b214e4a46" width="45%">
</p>


## XG Boost models

**Testing RMSE:**
FanDuel: 9.682575858687576<br>
DraftKings: 9.668348108287754<br>

**Live production RMSE (through March 3):**
Fanduel: 11.751239556366068
DraftKings: 12.205750130250914

**Features:**
(Rolling mean for the last 15 games)
fga: field-goal attempts
ast: assists
tptfgm: 3-point field goals made
fgm: field goals made
fta: free throws attempted
tptfga: 3-point field goals attempted
OffReb: offensive rebounds
ftm: free throws made
blk: blocked shots
DefReb: defensive rebounds
PlusMinus: team's point differential when player is on the court
stl: steals
pts: points scored
PF: personal fouls
TOV: Turnovers
usage: percentage of possessions in which player makes a shot attempt, makes free throw attempts or turns the ball over
mins_share: Percentage of overall team minutes played by that player, scaled to 240 (5 * 48, 5 players on floor for 48 minutes).
mins: Minutes played
mins_proj: Projected minutes played.







