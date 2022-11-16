"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-02
Description: Wrapper functions for loading parameters and variebles from csvs.
    
"""
import pandas as pd

def load_parameters(file='Parameters values in LHS Sports Match Sims.csv',
                    directory='C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models'):
    parameters_df = pd.read_csv(directory+'/'+file, index_col='Parameter')
    fixed_params = parameters_df['Fixed Value'].dropna()
    fixed_params = fixed_params.to_dict()
    parameters_df = parameters_df[parameters_df['Fixed Value'].isnull()]
    # no waning immunity or flows of people between clusters and vaccination groups.
    parameters_held_at_0 =  {param: 0
                             for param in
                             ['alpha', 'iota_{RA}', 'iota_{RTPCR}', 'nu_e', 'nu_b', 'nu_w']}
    fixed_params.update(parameters_held_at_0)
    return parameters_df, fixed_params

def load_repeated_sample(prevalence_file='Prevalence data.csv',
                         schedule_file='Fifa 2022 Group stages matches with venue capacity.csv',
                         directory='C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models'):
    prevalence_df = pd.read_csv(directory+'/'+prevalence_file, index_col='country')
    prevalence_dict = prevalence_df.prevalence.to_dict()
    schedule_df = pd.read_csv(directory+'/'+schedule_file)
    schedule_df['Team A prevalence'] = [prevalence_dict[county] for county in schedule_df['Team A']]
    schedule_df['Team B prevalence'] = [prevalence_dict[county] for county in schedule_df['Team B']]
    # For now we are removing matches where qatar plays.
    schedule_df = schedule_df[schedule_df['Team A']!='Qatar']
    schedule_df = schedule_df[schedule_df['Team B']!= 'Qatar']
    selection = ['Team A prevalence', 'Team B prevalence']#, 'Capacity']
    return schedule_df[selection]

