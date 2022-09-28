"""
Creation:
    Author: Martin Grunnill
    Date: 28/09/2022
Description: Test run of world cup match simulation.
    
"""
#%%
# import packages

import numpy as np
import json
import os.path
import pandas as pd

from simulations.sports_match_sim import SportMatchMGESimulation

# for now assume England vs Wales Ahmed Bin Ali stadium Tuesday 29th Nov
# Team a: England, Team B: Wales


stadium_capacity = 40000 # https://hukoomi.gov.qa/en/article/world-cup-stadiums

# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
host_population = 2930524
# From Qatars vaccination data enter populations
total_hosts_vaccinated = 2751485 # https://coronavirus.jhu.edu/region/qatar
hosts_effectively_vaccinated = 1870034 # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
qatar_cases_past_day = 733.429 # 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
host_cases_per_million = (qatar_cases_past_day/host_population)*1e6
UK_total_vaccinated = 50745901 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
UK_booster_doses = 40373987 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
team_A_proportion_effective = UK_booster_doses/UK_total_vaccinated
team_B_proportion_effective = team_A_proportion_effective
team_A_cases_per_million = 65.368  # UK as whole 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
team_B_cases_per_million = team_A_cases_per_million



sport_match_sim = SportMatchMGESimulation(host_population, host_cases_per_million,
                                          total_hosts_vaccinated, hosts_effectively_vaccinated,
                                          stadium_capacity,
                                          team_A_cases_per_million, team_B_cases_per_million)