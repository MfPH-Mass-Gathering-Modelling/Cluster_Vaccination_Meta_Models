"""
Creation:
    Author: Martin Grunnill
    Date: 2022-10-20
Description: Analysis of infection to detection ratio data from
https://ghdx.healthdata.org/sites/default/files/record-attached-files/HME_COVID_19_IES_2019_2021_RATIOS.zip
See paper at https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(22)00484-6/fulltext
"""
import pandas as pd

#%%
# Getting data frame
# location of zip file downloaded from https://ghdx.healthdata.org/sites/default/files/record-attached-files/HME_COVID_19_IES_2019_2021_RATIOS.zip
zip_file = 'C:/Users/mdgru/Downloads/HME_COVID_19_IES_2019_2021_RATIOS.zip'
# read file
data_df = pd.read_csv(zip_file)
#%% Selecting detections/infections
data_df.measure_name.unique()
data_df = data_df[data_df.measure_name=='Cumulative infection-detection ratio']
# change date to date
data_df['date'] = pd.to_datetime(data_df['date'])
data_df = data_df[data_df.date==data_df.date.max()]

#%%
# Qatari data

qatari_data = data_df[data_df.location_name=='Qatar']

#%%
# All other world cup nations
group_stage_df = pd.read_csv('Fifa 2022 Group stages matches with venue capacity.csv')
teams = group_stage_df['Team A'].unique().tolist()
teams += group_stage_df['Team B'].unique().tolist()
teams = set(teams)
teams.remove('Qatar')
other_data = data_df[data_df.location_name.isin(teams)]
len(teams)-len(other_data) # one missing
not_listed = list(set(teams) - set(other_data.location_name.tolist()))
# United States must be listed as USA or US
places = data_df.location_name.sort_values() # listed as USA
teams.add('USA')
teams.remove('United States')
other_data = data_df[data_df.location_name.isin(teams)]
upper_bounds = (other_data.value_lower.min()/100)**-1
lower_bounds = (other_data.value_upper.max()/100)**-1
other_data.value_mean.describe()
