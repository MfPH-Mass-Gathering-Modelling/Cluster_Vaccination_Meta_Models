"""
Creation:
    Author: Martin Grunnill
    Date: 2022-11-01
Description: Getting prevelance data for world cup teams.
    
"""
import copy
import pandas as pd
import datetime

schedule_df = pd.read_csv('Fifa 2022 Group stages matches with venue capacity.csv')
covid_data = pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
population_df = pd.read_csv('Population etimates world bank.csv',header=2, index_col='Country Name') # downloaded from https://data.worldbank.org/indicator/SP.POP.TOTL https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv

# need to change covid_data to datetime type
covid_data.date = pd.to_datetime(covid_data.date)

date_to = datetime.datetime(2022, 10, 31)
covid_data = covid_data[covid_data.date<=date_to]

#%%
# select data for only countries in the world cup
countries = schedule_df.Team_A.unique().tolist()
# looking at the data set (https://covid.ourworldindata.org/data/owid-covid-data.csv) new cases smoothed
# for England and Wales is pooled under United Kingdom
proxies = copy.deepcopy(countries)
proxies.append('United Kingdom')
proxies.remove('England')
proxies.remove('Wales')
proxies = sorted(proxies)
covid_data = covid_data[covid_data.location.isin(proxies)]
# sense check to make sure we have selected the right places
selected_proxies = covid_data.location.unique()
len(proxies)==len(selected_proxies)
#%% Selecting most recent available data for new_cases_smoothed_per_million
# remove missing data
covid_data.new_cases_smoothed_per_million = covid_data.new_cases_smoothed_per_million.replace({0:None})
covid_data = covid_data[pd.notnull(covid_data.new_cases_smoothed_per_million)]
prevelance_records = []
for country in countries:
    if country in ['England', 'Wales']:
        proxy = 'United Kingdom'
    else:
        proxy = country
    # select proxie
    location_data = covid_data[covid_data.location==proxy]
    # latest date for which we have information
    latest_date = location_data.date.max()
    latest_date_data = location_data[location_data.date==latest_date]
    cases_smoothed = latest_date_data.new_cases_smoothed.iloc[0]
    if proxy=='South Korea':
        population = population_df.loc['Korea, Rep.', '2021']
    elif proxy == 'Iran':
        population = population_df.loc['Iran, Islamic Rep.', '2021']
    else:
        population = population_df.loc[proxy,'2021']

    entry = {'country': country,
             'proxy': proxy,
             'date': latest_date,
             'prevalence': cases_smoothed/population,
             }

    prevelance_records.append(entry)

prevelance_df = pd.DataFrame(prevelance_records)
prevelance_df.to_csv('Prevalence data.csv', index=False)
# #%% Adding data on infection to detection ratio
#
# # Getting data frame
# # location of zip file downloaded from https://ghdx.healthdata.org/sites/default/files/record-attached-files/HME_COVID_19_IES_2019_2021_RATIOS.zip
# zip_file = 'C:/Users/mdgru/Downloads/HME_COVID_19_IES_2019_2021_RATIOS.zip'
# # read file
# detection_raio_df = pd.read_csv(zip_file)
# #%% Selecting detections/infections
# detection_raio_df.measure_name.unique()
# detection_raio_df = detection_raio_df[detection_raio_df.measure_name=='Cumulative infection-detection ratio']
# # change date to date
# detection_raio_df['date'] = pd.to_datetime(detection_raio_df['date'])
# detection_raio_df = detection_raio_df[detection_raio_df.date==detection_raio_df.date.max()]
# detection_raio_df.location_name = detection_raio_df.location_name.replace({'USA':'United States',
#                                                                            'UK':'United Kingdom'})


