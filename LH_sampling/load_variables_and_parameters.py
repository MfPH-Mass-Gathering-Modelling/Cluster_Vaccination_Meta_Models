"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-02
Description: Wrapper functions for loading parameters and variebles from csvs.
    
"""
import pandas as pd

def load_parameters(file='Parameters values in LHS Sports Match Sims.csv'):
    parameters_df = pd.read_csv(file, index_col='Symbol')
    fixed_params = parameters_df['Fixed Value'].dropna()
    fixed_params = fixed_params.to_dict()
    parameters_df = parameters_df[parameters_df['Fixed Value'].isnull()]
    return parameters_df, fixed_params

def load_repeated_sample(prevalence_file='Prevalence data.csv',
                         schedule_file='Fifa 2022 Group stages matches with venue capacity.csv'):
    prevalence_df = pd.read_csv(prevalence_file, index_col='country')
    prevalence_dict = prevalence_df.prevalence.to_dict()
    schedule_df = pd.read_csv(schedule_file)
    schedule_df['Team A prevalence'] = [prevalence_dict[county] for county in schedule_df.Team_A]
    schedule_df['Team B prevalence'] = [prevalence_dict[county] for county in schedule_df.Team_B]
    # For now we are removing matches where qatar plays.
    schedule_df = schedule_df[schedule_df['Team A'!='Qatar']]
    schedule_df = schedule_df[schedule_df['Team B' != 'Qatar']]
    return schedule_df

