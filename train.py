import glob

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

PRED_COLS = ['AST', 'BLK', 'DREB', 'FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM', 'OREB', 'PF', 'STL', 'TO']

REGULAR_SEASON_DURATION = {
    2021: ('2021-10-19', '2022-04-10'),
    2020: ('2020-12-22', '2021-05-16'),
    2018: ('2018-10-16', '2019-04-10'),
    2017: ('2017-10-17', '2018-04-11'),
    2016: ('2016-10-25', '2017-04-12'),
    2015: ('2015-10-27', '2016-04-13'),
    2014: ('2014-10-28', '2015-04-15'),
    2013: ('2013-10-29', '2014-04-16'),
    2012: ('2012-10-30', '2013-04-17'),
}

PREDICTION_STATS = ['AST_home', 'BLK_home', 'DREB_home', "FG3A_home", "FGA_home", "FTA_home", 'OREB_home', 'STL_home', "TO_home",
                    'AST_away', 'BLK_away', 'DREB_away', "FG3A_away", "FGA_away", "FTA_away", 'OREB_away', 'STL_away', "TO_away"]
#%%

def get_season_games(season=2021):
    games = dfs['games'].copy()
    games = games[['TEAM_ID_home', 'TEAM_ID_away', 'HOME_TEAM_WINS', 'GAME_ID', 'GAME_DATE_EST']]

    regular_games = games[(REGULAR_SEASON_DURATION[season][0] <= pd.to_datetime(games['GAME_DATE_EST'])) & (
                pd.to_datetime(games['GAME_DATE_EST']) <= REGULAR_SEASON_DURATION[season][1])]

    return regular_games


def get_team_boxscore(team_id=1610612738, is_home=True):
    games = get_season_games(2021)
    games_details = dfs['games_details'].copy()
    # join games to games_details for date information
    games_details = games_details.merge(games, on='GAME_ID')

    # add additional column indicating whether team is home or not
    games_details['is_home'] = games_details['TEAM_ID'] == games_details['TEAM_ID_home']

    # select data from team at home or away
    games_details = games_details[(games_details['TEAM_ID'] == team_id) & (games_details['is_home'] == is_home)][
        PRED_COLS + ['GAME_ID']]

    # sum over all players for each game and then average over all games
    boxscore = games_details.groupby(['GAME_ID']).sum().mean().to_frame().T

    return boxscore


def get_team_boxscore_by_position(team_id=1610612738, is_home=True, start_position='F'):
    games = get_season_games(2021)
    games_details = dfs['games_details'].copy()
    # join games to games_details for date information
    games_details = games_details.merge(games, on='GAME_ID')

    # add additional column indicating whether team is home or not
    games_details['is_home'] = games_details['TEAM_ID'] == games_details['TEAM_ID_home']

    # select data from team at home or away
    games_details = games_details[
        (games_details['TEAM_ID'] == team_id) & (games_details['START_POSITION'] == start_position) & (
                    games_details['is_home'] == is_home)][PRED_COLS + ['GAME_ID', 'START_POSITION']]

    # sum over all players for each game and then average over all games
    boxscore = games_details.groupby(['GAME_ID']).sum().mean().to_frame().T

    return boxscore


def get_season_boxscore(season=2021):
    games = get_season_games(season)
    games_details = dfs['games_details'].copy()

    # add games details to regular season games
    games_details = games_details.merge(games, on='GAME_ID')

    # add additional column indicating whether team is home or not
    games_details['is_home'] = games_details['TEAM_ID'] == games_details['TEAM_ID_home']

    games_details = games_details[PRED_COLS + ['GAME_ID', 'TEAM_ID', 'is_home']]
    # group by game and team and sum stats
    games_details_agg = games_details.groupby(['GAME_ID', 'TEAM_ID']).sum()
    # reset is_home col to boolean value
    games_details_agg['is_home'] = games_details_agg['is_home'] > 0
    # group by team and is_home
    #games_details_agg = games_details_agg.groupby(['TEAM_ID', 'is_home']).mean()
    # only keep prediction columns
    games_details_agg = games_details_agg.reset_index()
    #games_details_agg = games_details_agg[PRED_COLS]
    return games_details_agg


def get_train_data(season=2021):
    season_games = get_season_games(season)
    games_details = get_season_boxscore(season)

    # add stats of home teams
    train_data = season_games.merge(games_details[games_details['is_home']].rename(columns={'GAME_ID':'GAME_id'}).drop(columns='is_home'),
                                    left_on=['TEAM_ID_home', 'GAME_ID'], right_on=['TEAM_ID', 'GAME_id'])
    # rename stats cols
    train_data = train_data.rename(columns={col: f"{col}_home" for col in PRED_COLS})
    # add stats of away teams
    train_data = train_data.merge(games_details[~games_details['is_home']].rename(columns={'GAME_ID':'GAME_id'}).drop(columns='is_home'),
                                  left_on=['TEAM_ID_away', 'GAME_ID'], right_on=['TEAM_ID', 'GAME_id'])
    # rename stats cols
    train_data = train_data.rename(columns={col: f"{col}_away" for col in PRED_COLS})

    # only return feature and target columns
    return train_data[[f"{col}_home" for col in PRED_COLS] + [f"{col}_away" for col in PRED_COLS] + ['HOME_TEAM_WINS']]
def calculate_advanced_stat(train_data):
    train_data_advanced = train_data.copy()
    for index, row in train_data_advanced.iterrows():
        # points
        points_home = 2 * (row["FGM_home"] + 0.5 * row["FG3M_home"]) + row["FTM_home"]
        points_away = 2 * (row["FGM_away"] + 0.5 * row["FG3M_away"]) + row["FTM_away"]
        train_data_advanced.loc[index, "Points_home"] = points_home
        train_data_advanced.loc[index, "Points_away"] = points_away

        # possession = .5 * (FGA + .475 * FTA - ORB + TOV)
        possessions_home = 0.5 * (row["FGA_home"] + 0.475 * row["FTA_home"] - row["OREB_home"] + row["TO_home"])
        possessions_away = 0.5 * (row["FGA_away"] + 0.475 * row["FTA_away"] - row["OREB_away"] + row["TO_away"])
        train_data_advanced.loc[index, "Possesions_home"] = possessions_home
        train_data_advanced.loc[index, "Possesions_away"] = possessions_away

        # Effective Field Goal Percentage (eFG%) = (FG + .5 * 3P) / FGA
        train_data_advanced.loc[index, "EFG%_home"] = (row["FGM_home"] + 0.5 * row["FG3M_home"]) / row["FGA_home"]
        train_data_advanced.loc[index, "EFG%_away"] = (row["FGM_away"] + 0.5 * row["FG3M_away"]) / row["FGA_away"]

        # True Shooting Percentage (TS%) = Pts / (2 * (FGA + .475 * FTA))
        train_data_advanced.loc[index, "TS%_home"] = points_home / (2 * (row["FGA_home"] + 0.475 * row["FTA_home"]))
        train_data_advanced.loc[index, "TS%_away"] = points_away / (2 * (row["FGA_away"] + 0.475 * row["FTA_away"]))

        # Turnover Percentage (TOV%) = TOV / (FGA + .475*FTA + AST + TOV)
        train_data_advanced.loc[index, "TOV%_home"] = row["TO_home"] / (
                    row["FGA_home"] + 0.475 * row["FTA_home"] + row["TO_home"])
        train_data_advanced.loc[index, "TOV%_away"] = row["TO_away"] / (
                    row["FGA_away"] + 0.475 * row["FTA_away"] + row["TO_away"])

        # Offensive Rating (OR) = 100 / (TmPoss + OppPoss) * Pts
        train_data_advanced.loc[index, "OR_home"] = (100 * points_home) / (possessions_home + possessions_away)
        train_data_advanced.loc[index, "OR_away"] = (100 * points_away) / (possessions_home + possessions_away)

        # Defensive Rating (DR) = 100 / (TmPoss + OppPoss) * OppPts
        train_data_advanced.loc[index, "DR_home"] = (100 * points_away) / (possessions_home + possessions_away)
        train_data_advanced.loc[index, "DR_away"] = (100 * points_home) / (possessions_home + possessions_away)

    return train_data_advanced

def oliver_factors(train_data):
    oliver = pd.DataFrame(
        columns=['shooting_home', 'to_home', 'reb_home', 'ft_home', 'shooting_away', 'to_away', 'reb_away', 'ft_away'])
    oliver['shooting_home'] = (train_data['FGM_home'] + 0.5 * train_data['FG3M_home']) / \
                              train_data['FGA_home']
    oliver['to_home'] = (train_data['TO_home']) / (
                train_data["FGA_home"] + 0.44 * train_data["FTA_home"] + train_data[
            "TO_home"])
    oliver['reb_home'] = (train_data['OREB_home']) / (
                train_data["OREB_home"] + train_data["DREB_home"])
    oliver['ft_home'] = (train_data['FTM_home']) / (train_data["FGA_home"])

    oliver['shooting_away'] = (train_data['FGM_away'] + 0.5 * train_data['FG3M_away']) / \
                              train_data['FGA_away']
    oliver['to_away'] = (train_data['TO_away']) / (
                train_data["FGA_away"] + 0.44 * train_data["FTA_away"] + train_data[
            "TO_away"])
    oliver['reb_away'] = (train_data['OREB_away']) / (
                train_data["OREB_away"] + train_data["DREB_away"])
    oliver['ft_away'] = (train_data['FTM_away']) / (train_data["FGA_away"])

    oliver_y = train_data["HOME_TEAM_WINS"]

    return oliver, oliver_y

def normal_training(train_data):
    y = train_data["HOME_TEAM_WINS"]
    X = train_data[PREDICTION_STATS]
    return X, y

def difference(train_data):
    y = train_data["HOME_TEAM_WINS"]
    X = train_data[PREDICTION_STATS]

    columns = X.columns
    column_groups = {}

    for column in columns:
        prefix = column[:3]
        if prefix not in column_groups:
            column_groups[prefix] = []
        column_groups[prefix].append(column)

    # Calculate the differences between columns in each group
    for prefix, group in column_groups.items():
        for i in range(1, len(group)):
            column_diff = f"{group[i]}_diff"
            X[column_diff] = X[group[i]] - X[group[i - 1]]

    X = X.drop(list(X.columns[:-9]), axis=1)

    return X, y

def rescaling(X, mode='min-max'):
    if mode == 'min-max':
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(X)
    if mode == 'normalizer':
        scaler = Normalizer()
        rescaledX = scaler.fit_transform(X)
    if mode == 'standard':
        scaler = StandardScaler()
        rescaledX = scaler.fit_transform(X)
    return rescaledX

files = glob.glob('*.csv')

dfs = {}
for file_path in files:
    dfs[file_path.split(".")[-2].split("/")[-1]] = pd.read_csv(file_path)

train_data = []
for season in REGULAR_SEASON_DURATION:
    train_data.append(get_train_data(season))

train_data = pd.concat(train_data, axis=0)

train_data_advanced = calculate_advanced_stat(train_data)


'''---------------- Normal training --------------'''

param = {
    'boosting_type': ['gbdt', 'dart'],
    'min_child_samples': [20, 100, 300],
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.01, 0.01],
}

X, y = normal_training(train_data_advanced)

rescaledX = rescaling(X, 'standard')

model = lgb.LGBMClassifier(max_depth= -1, random_state=0, objective='binary')

clf_normal = GridSearchCV(model, param)
clf_normal.fit(rescaledX, y)

print('Boxscore input acc: ', np.max(clf_normal.cv_results_['mean_test_score']))

'''---------------- Difference training --------------'''

param = {
    'boosting_type': ['gbdt', 'dart'],
    'min_child_samples': [20, 100, 300],
    'n_estimators': [50, 100, 500],
    'learning_rate': [0.01, 0.01],
}

X, y = difference(train_data_advanced)

rescaledX = rescaling(X, 'standard')

model = lgb.LGBMClassifier(max_depth= -1, random_state=0, objective='binary')

clf_diff = GridSearchCV(model, param)
clf_diff.fit(rescaledX, y)

print('Difference input acc: ', np.max(clf_diff.cv_results_['mean_test_score']))

'''---------------- Oliver regression --------------'''

X, y = oliver_factors(train_data_advanced)

model = LogisticRegression(random_state=0, penalty='none', fit_intercept=False)

oliver_ac = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print('Oliver acc:', np.mean(oliver_ac))


exit(0)