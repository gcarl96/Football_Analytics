import sqlite3
import pandas as pd
import numpy as np
import json
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from random import randint
from sklearn.preprocessing import StandardScaler


class DataManager():

    def __init__(self):
        #overall_rating, finishing, short_passing, dribbling, ball_control, vision, standing_tackle, sliding_tackle, marking, positioning, stamina, sprint_speed
        self.conn = sqlite3.connect('database.sqlite')
        self.attributes = ['overall_rating', 'short_passing', 'dribbling', 'ball_control','positioning']
        self.num_attrs = 1
        self.player_attrs_df = self.sql_to_pd('''SELECT player_api_id, overall_rating, short_passing, dribbling, ball_control, positioning, standing_tackle, date FROM Player_Attributes''')#.dropna()
        

    def sql_to_pd(self,query):
        return pd.read_sql_query(query,self.conn)

    def conv_to_prob(self,frame):
        frame['B365H_Prob'] = frame['B365H'].apply(lambda x: (1/x))
        frame['B365D_Prob'] = frame['B365D'].apply(lambda x: (1/x))
        frame['B365A_Prob'] = frame['B365A'].apply(lambda x: (1/x))
        frame['odds_total'] = frame['B365H_Prob'] + frame['B365D_Prob'] + frame['B365A_Prob']
        frame['B365H_Prob'] = frame['B365H_Prob'] / frame['odds_total']
        frame['B365D_Prob'] = frame['B365D_Prob'] / frame['odds_total']
        frame['B365A_Prob'] = frame['B365A_Prob'] / frame['odds_total']
        frame['odds'] = frame[['B365H_Prob','B365D_Prob','B365A_Prob']].values.tolist()

    def calc_results(self,frame):
        frame['goal_diff'] = frame['home_team_goal'] - frame['away_team_goal']
        frame['result'] = frame['goal_diff'].apply(lambda x: [1,0,0] if x > 0 else ([0,1,0] if x == 0 else [0,0,1]))

    def get_section_matches(self,matches, season, min_pos, max_pos):
        prev_season_table = pd.read_csv('league_tables/'+season+'.csv')
        teams_in_section = prev_season_table[(prev_season_table["Pos"] > min_pos) & (prev_season_table["Pos"] < max_pos)]['Team'].values
        section_matches = matches[(matches['home_team_name'].isin(teams_in_section)) | (matches['away_team_name'].isin(teams_in_section))]
        return section_matches

    def get_player_attributes(self,matches_df, player_and_attribute_df, is_home, player_id):
        matches_df.rename(columns={'date': 'date_match'}, inplace=True)
        matches_df.rename(columns={'id': 'id_match'}, inplace=True)

        right_cond = 'home_player_' + str(player_id) if is_home else 'away_player_' + str(player_id)
        matches_with_player = pd.merge(matches_df, player_and_attribute_df, how='inner', left_on=right_cond, right_on="player_api_id")

        # Filter out all ratings with dates greater than match date; sfxs[0] is appended because the identifier depends on it
        matches_with_past_ratings = matches_with_player[(matches_with_player['date'] <= matches_with_player['date_match'])]
        result = matches_with_past_ratings.sort_values('date', ascending=False).drop_duplicates(['id_match'])

        columns_to_drop = list(filter(lambda cln: False if (cln == 'id_match') else True, matches_df.columns.values))
        columns_to_drop = list(map(lambda cln: cln + '_match' if (cln == 'date') else cln, columns_to_drop))
        result = result.drop(columns_to_drop, 1)

        return result

    def build_match_data(self,match_frame, matches_layout, matches_flat, matches_attrs, match_results, match_odds, match_teams, regression):
        match_layout = np.zeros(shape=(9,22,self.num_attrs))
        match_flat = np.zeros(shape=(22,1))
        match_attrs = np.zeros(shape=(14,1))

        home_gk_rating = 0
        home_def_ratings = []
        home_mid_ratings = []
        home_off_ratings = []
        home_prev_league_pos = 0
        home_ages = []
        away_gk_rating = 0
        away_def_ratings = []
        away_mid_ratings = []
        away_off_ratings = []
        away_prev_league_pos = 0
        away_ages = []
        for i in range(11):

            home_x = int(match_frame['home_player_X'+str(i+1)]-1)
            if home_x == 0:
                home_x = 4
            home_y = int(match_frame['home_player_Y'+str(i+1)]-1)
            home_attrs = self.player_attributes_home[i][self.player_attributes_home[i]['id_match'] == match_frame['id_match']][['overall_rating', 'overall_rating_scaled']].values[0]
            match_layout[home_x][home_y] = home_attrs[1]
            match_flat[i] = home_attrs[0]
            #gk
            if home_y == 0:
                home_gk_rating = home_attrs[0]
            #def
            elif home_y < 4: 
                home_def_ratings.append(home_attrs[0])
            #mid
            elif home_y < 8:
                home_mid_ratings.append(home_attrs[0])
            #off
            else:
                home_off_ratings.append(home_attrs[0])

            away_x = int(9-match_frame['away_player_X'+str(i+1)])
            if away_x == 8:
                away_x = 4
            away_y = int(22-match_frame['away_player_Y'+str(i+1)])
            away_attrs = self.player_attributes_away[i][self.player_attributes_away[i]['id_match'] == match_frame['id_match']][['overall_rating', 'overall_rating_scaled']].values[0]
            match_layout[away_x][away_y] = away_attrs[1]
            match_flat[i+11] = away_attrs[0]
            #gk
            if away_y == 21:
                away_gk_rating = away_attrs[0]
            #def
            elif away_y > 17: 
                away_def_ratings.append(away_attrs[0])
            #mid
            elif away_y > 13:
                away_mid_ratings.append(away_attrs[0])
            #off
            else:
                away_off_ratings.append(away_attrs[0])
                    
        # match_attrs = [np.mean(home_gk_rating),
        #                         np.mean(home_def_ratings),
        #                         np.mean(home_mid_ratings),
        #                         np.mean(home_off_ratings),
        #                         np.mean(away_gk_rating),
        #                         np.mean(away_def_ratings),
        #                         np.mean(away_mid_ratings),
        #                         np.mean(away_off_ratings),
        match_attrs =          [match_frame['home_prev_league_pos'],
                                match_frame['away_prev_league_pos'],
                                match_frame['home_team_form'],
                                match_frame['away_team_form'],
                                match_frame['B365H'],
                                match_frame['B365D'],
                                match_frame['B365A']]
        # match = match_frame[['B365H','B365D','B365A']].values

        
        match_flat = match_flat.flatten()

        matches_layout[match_frame['match_num']] = match_layout
        matches_flat[match_frame['match_num']] = match_flat
        matches_attrs[match_frame['match_num']] = match_attrs
        match_odds[match_frame['match_num']] = match_frame[['B365H_Prob','B365D_Prob','B365A_Prob']].values#match_frame[['B365H','B365D','B365A']].values

        if not regression:
            if match_frame['home_team_goal'] > match_frame['away_team_goal']:
                match_results[match_frame['match_num']] = 0
            elif match_frame['home_team_goal'] < match_frame['away_team_goal']:
                match_results[match_frame['match_num']] = 2
            else:
                match_results[match_frame['match_num']] = 1
        else:
            match_results[match_frame['match_num']] = match_frame['home_team_goal'] - match_frame['away_team_goal']
        match_teams[match_frame['match_num']] = match_frame[['home_team_name','away_team_name']]
        # match_results[match_frame['match_num']] = match_frame['home_team_goal'] - match_frame['away_team_goal']

    def buildData(self,data_seasons, regression):

        inputs = {
            "layouts" : [],
            "flats" : [],
            "attrss" : []
        }
        results = []
        odds = []
        teams = []

        scaler = StandardScaler()
        self.player_attrs_df['overall_rating_scaled'] = scaler.fit_transform(self.player_attrs_df['overall_rating'].values.reshape(-1,1))

        # leagues = sql_to_pd('''SELECT DISTINCT name FROM League''')
        leagues = ["England Premier League"]
        for i in range(len(leagues)):
            print("----")
            briers = []
            rpss = []
            accs = []
            # league = leagues.iloc[i]['name']
            league = leagues[i]
            print(league)
            seasons = self.sql_to_pd('''SELECT DISTINCT Match.season FROM Match INNER JOIN League ON (League.id = Match.league_id) WHERE League.Name = "{0}"'''.format(league))
            print(f"Season: {seasons}")
            for j in range(len(seasons)):
            # for j in [5,6]:
                season = seasons.iloc[j]['season']
                if not season in data_seasons:
                    print("Season not found")
                    continue
                prev_season = seasons.iloc[j-1]['season']
                print(season)
                query = '''SELECT Match.id, date, home_team_goal, away_team_goal, B365H, B365D, B365A, th.team_long_name as home_team_name, ta.team_long_name as away_team_name,
                home_player_11,home_player_1,home_player_2,home_player_3,home_player_4,home_player_5,home_player_6,home_player_7,home_player_8,home_player_9,home_player_10,
                away_player_11,away_player_1,away_player_2,away_player_3,away_player_4,away_player_5,away_player_6,away_player_7,away_player_8,away_player_9,away_player_10,
                home_player_X11,home_player_Y11,home_player_X1,home_player_Y1,home_player_X2,home_player_Y2,home_player_X3,home_player_Y3,home_player_X4,home_player_Y4,home_player_X5,home_player_Y5,home_player_X6,home_player_Y6,home_player_X7,home_player_Y7,home_player_X8,home_player_Y8,home_player_X9,home_player_Y9,home_player_X10,home_player_Y10,
                away_player_X11,away_player_Y11,away_player_X1,away_player_Y1,away_player_X2,away_player_Y2,away_player_X3,away_player_Y3,away_player_X4,away_player_Y4,away_player_X5,away_player_Y5,away_player_X6,away_player_Y6,away_player_X7,away_player_Y7,away_player_X8,away_player_Y8,away_player_X9,away_player_Y9,away_player_X10,away_player_Y10
                FROM Match
                INNER JOIN League
                ON (League.id = Match.league_id) 
                JOIN Team th ON (th.team_api_id = home_team_api_id)
                JOIN Team ta ON (ta.team_api_id = away_team_api_id)
                WHERE League.name = '{0}' AND Match.season = '{1}'
                ORDER BY date;'''.format(league, season)

                matches_df = self.sql_to_pd(query).dropna().reset_index()
                # matches_df.date = matches_df.date.apply(lambda x: x.split(' ')[0])
                # matches_df = matches_df.sort_values(by='date')
                matches_df['match_num'] = matches_df.index
                prev_season_table = pd.read_csv('league_tables/'+str(league.replace(" ", "_"))+"_"+str(prev_season.replace("/",":"))+'.csv')
                
                self.conv_to_prob(matches_df)
                self.calc_results(matches_df)
                matches_df['home_prev_league_pos'] = matches_df['home_team_name'].apply(lambda x: prev_season_table[prev_season_table["Team"] == x]["Pos"].values[0] if x in prev_season_table["Team"].values else 21)
                matches_df['away_prev_league_pos'] = matches_df['away_team_name'].apply(lambda x: prev_season_table[prev_season_table["Team"] == x]["Pos"].values[0] if x in prev_season_table["Team"].values else 21)

                conditions = [ (matches_df['home_team_goal'] > matches_df['away_team_goal']),
                                (matches_df['home_team_goal'] < matches_df['away_team_goal'])]
                choices = [3, 0]
                matches_df['home_team_points'] = np.select(conditions, choices, default=1)
                choices = [0, 3]
                matches_df['away_team_points'] = np.select(conditions, choices, default=1)

                # prev_games = 10
                # for j in range(len(matches_df)):
                #     # print(matches_df.loc[j,'match_num'])
                #     home_team = matches_df.loc[j, 'home_team_name']
                #     away_team = matches_df.loc[j, 'away_team_name']
                #     matches_df.loc[j, 'home_team_form'] = -1
                #     matches_df.loc[j, 'away_team_form'] = -1
                #     if j > prev_games:
                #         home_team_games = matches_df[(matches_df['match_num'] < matches_df.loc[j, 'match_num']) & ((matches_df['home_team_name'] == home_team) | (matches_df['away_team_name'] == home_team))]
                #         matches_df.loc[j, 'home_team_form'] = np.sum(np.concatenate([home_team_games[home_team_games['home_team_name'] == home_team]['home_team_points'].values, home_team_games[home_team_games['away_team_name'] == home_team]['away_team_points'].values])[-prev_games:])
                #         away_team_games = matches_df[(matches_df['match_num'] < matches_df.loc[j, 'match_num']) & ((matches_df['home_team_name'] == away_team) | (matches_df['away_team_name'] == away_team))]
                #         matches_df.loc[j, 'away_team_form'] = np.sum(np.concatenate([away_team_games[away_team_games['home_team_name'] == away_team]['home_team_points'].values, away_team_games[away_team_games['away_team_name'] == away_team]['away_team_points'].values])[-prev_games:])

                print(matches_df.head())

                matches_df.dropna(inplace=True)
                self.player_attributes_home = [self.get_player_attributes(matches_df, self.player_attrs_df, True, i) for i in range(1,12)]
                self.player_attributes_away = [self.get_player_attributes(matches_df, self.player_attrs_df, False, i) for i in range(1,12)]

                matches_layout = np.empty(len(matches_df), dtype=object)
                matches_flat = np.empty(len(matches_df), dtype=object)
                matches_attrs = np.empty(len(matches_df), dtype=object)
                match_results = np.empty(len(matches_df), dtype=object)
                match_odds = np.empty(len(matches_df), dtype=object)
                match_teams = np.empty(len(matches_df), dtype=object)

                build_all_data = lambda x: self.build_match_data(x, matches_layout, matches_flat, matches_attrs, match_results, match_odds, match_teams, regression)

                build_square_data = lambda x: self.build_match_data(x,"square", match_layouts, match_results, match_odds, match_teams, regression)
                build_flat_data = lambda x: self.build_match_data(x,"flat", match_layouts, match_results, match_odds, match_teams, regression)
                build_reg_features_data = lambda x: self.build_match_data(x,"reg_features", match_layouts, match_results, match_odds, match_teams, regression)

                matches_df.apply(build_all_data,1)

                # if shape == "square":
                #     matches_df.apply(build_square_data,1)
                # elif shape == "flat":
                #     matches_df.apply(build_flat_data,1)
                # else:
                #     matches_df.apply(build_reg_features_data,1)
                print("Successful: " + str(len(matches_layout)) + " " + str(len(matches_flat)) + " " + str(len(matches_attrs)))
                # print(brier_score(match_odds,match_results))
                # match_layouts = np.array([np.array(layout) for layout in match_layouts if not np.isnan(np.sum(layout))])


                valid = [i for i in range(len(matches_layout)) if not np.isnan(np.sum(matches_layout[i])) and not np.isnan(np.sum(matches_flat[i])) and not np.isnan(np.sum(matches_attrs[i]))]
                matches_layout = np.array([np.array(layout) for layout in matches_layout[valid]])
                matches_flat = np.array([np.array(layout) for layout in matches_flat[valid]])
                matches_attrs = np.array([np.array(layout) for layout in matches_attrs[valid]])
                match_results = np.array([np.array(result) for result in match_results[valid]])
                match_odds = np.array([np.array(odds) for odds in match_odds[valid]])
                match_teams = np.array([np.array(teams) for teams in match_teams[valid]])

                inputs['layouts'] += [matches_layout]
                inputs['flats'] += [matches_flat]
                inputs['attrss'] += [matches_attrs]
                results += [match_results]
                odds += [match_odds]
                teams += [match_teams]

                # np.save("db_formatted_data/5_attrs/"+(shape)+"/"+data_filename+"_layouts_"+str(num_attrs), match_layouts)
                # np.save("db_formatted_data/5_attrs/"+(shape)+"/"+data_filename+"_results_"+str(num_attrs), match_results)
                # np.save("db_formatted_data/5_attrs/"+(shape)+"/"+data_filename+"_odds_"+str(num_attrs), match_odds)

        inputs['layouts'] = np.concatenate(inputs['layouts'])
        inputs['flats'] = np.concatenate(inputs['flats'])
        inputs['attrss'] = np.concatenate(inputs['attrss'])
        return inputs, np.concatenate(results), np.concatenate(odds), np.concatenate(teams)


