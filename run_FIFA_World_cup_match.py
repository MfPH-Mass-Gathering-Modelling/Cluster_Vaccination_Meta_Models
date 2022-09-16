"""
Creation:
    Author: Martin Grunnill
    Date: 14/09/2022
Description:
    
"""
#%%
# import packages

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os.path
import copy
import pandas as pd


# %%
# import model
from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel
# import function for calculating beta from R_0
from CVM_models.Reproductive_Number.Beta_and_Reproduction import MGE_beta_no_vaccine_1_cluster
# import seeding methods
from seeding.multinomail_seeding import MultnomialSeeder
# import transfer class
from transfer_at_t.transfer_handling import TranfersAtTsScafold
from cleaning_up_results.results_to_dfs import results_array_to_df, results_df_pivoted

guess_at_case_to_infection_ratio = 10


#%%
# get model meta population structure
structures_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
                  'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
                  'CVM_models/Model meta population structures/')
with open(structures_dir + "World cup MGE.json", "r") as json_file:
    group_info=json_file.read()

group_info = json.loads(group_info)


#%%
# Initialise model


world_cup_model = MassGatheringModel(group_info)

#%%
# If available attach Jacobian. Note this has to avaluated for every different meta-population structure.
# See CVM_models/pure_python_models/deriving_MG_model_jacobian.py
pure_py_mods_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/Mass Gathering work/'+
                    'Compartment based models/Cluster_Vaccination_Meta_Models/CVM_models/pure_python_models/')

json_file = pure_py_mods_dir+"World_cup_MGEmodel_jacobian.json"

if os.path.exists(json_file):
    world_cup_model.load_dok_jacobian(json_file)

# %%
# Test parameter values
kappa = 0.25
params_for_deriving_beta = {
    # parameter values - Vaccination parameters
    # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
    'epsilon_3': 1,
    'p_s': 0.724,
    'p_d': 1/guess_at_case_to_infection_ratio,
    'p_h': 0.10,
    'gamma_A_1': 0.139*2,
    'gamma_A_2': 0.139*2,
    'gamma_I_1': 0.1627*2,
    'gamma_I_2': 0.1627*2,
    'theta': 0.342,
    'kappa_D': kappa,
}

# Rest
R_0 = 3
beta = MGE_beta_no_vaccine_1_cluster(R_0=R_0, **params_for_deriving_beta)

nu = 0 # Assume no new vaccinations.
beta_for_test_positive = beta*kappa
test_params = {**params_for_deriving_beta,
               # Test parameter values - Vaccination parameters
               # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
               # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
               # For now we will assume that the 3rd dose is effective as the 2nd dose before wanning immunity.
               'beta_host_spectators': beta,
               'beta_host_spectators_LFDpositive':beta_for_test_positive,
               'beta_host_spectators_PCRpositive':beta_for_test_positive,
               'beta_host_spectators_PCRwaiting': beta,
               'beta_hosts': beta,
               'beta_hosts_LFDpositive': beta_for_test_positive,
               'beta_hosts_PCRpositive': beta_for_test_positive,
               'beta_hosts_PCRwaiting': beta,
               'beta_hosts': beta,
               'beta_team_A_supporters': beta,
               'beta_team_A_supporters_LFDpositive': beta_for_test_positive,
               'beta_team_A_supporters_PCRpositive': beta_for_test_positive,
               'beta_team_A_supporters_PCRwaiting': beta,
               'beta_team_B_supporters': beta,
               'beta_team_B_supporters_LFDpositive': beta_for_test_positive,
               'beta_team_B_supporters_PCRpositive': beta_for_test_positive,
               'beta_team_B_supporters_PCRwaiting': beta,
               'epsilon_1': 0.5988023952,
               'epsilon_2': 0.4291845494,
               'gamma_H': 0.0826446281,
               'h_unvaccinated':0,
               'h_effective': 0.957,
               'h_waned':0.815,
               'l_effective':0.7443333333,
               'l_unvaccinated': 0,
               'l_waned': 0.2446666667,
               'nu_1':nu,
               'nu_b':nu,
               'nu_w':nu,
               'omega_G':1, # PCR result delay
               's_effective': 0.7486666667,
               's_unvaccinated': 0,
               's_waned': 0.274,
               'tau_A': 0, # Testing is performed by an event.
               'tau_G': 0, # Testing is performed by an event.
               'alpha':0, # considering the timeframe of this model alpha is not needed.
               'iota': 0 # model runtime is to short to consider end of isolation.
               }

world_cup_model.parameters = test_params

#%%
infection_branches = {'asymptomatic':{'E':'epsilon_1',
                                      'G_A':'epsilon_2',
                                      'P_A':'epsilon_3',
                                      'M_A':'gamma_A_1',
                                      'F_A':'gamma_A_2'},
                      'symptomatic': {'E': 'epsilon_1',
                                      'G_I': 'epsilon_2',
                                      'P_I': 'epsilon_3',
                                      'M_I': 'gamma_I_1',
                                      'F_I': 'gamma_I_2'},
                      'detected': {'E': 'epsilon_1',
                                   'G_I': 'epsilon_2',
                                   'P_I': 'epsilon_3',
                                   'M_D': 'gamma_I_1',
                                   'F_D': 'gamma_I_2'},
                      'hospitalised': {'E': 'epsilon_1',
                                       'G_I': 'epsilon_2',
                                       'P_I': 'epsilon_3',
                                       'H': 'gamma_H'}}

# setting up seeding class
world_cup_seeder = MultnomialSeeder(infection_branches)
infection_branch_proportions = {'asymptomatic': 1-test_params['p_s'],
                                'symptomatic': test_params['p_s']*(1-test_params['p_d']),
                                'detected': test_params['p_s']*test_params['p_d']*(1-test_params['p_h']),
                                'hospitalised': test_params['p_s']*test_params['p_d']*test_params['p_h']}


#%%
# Setting up population
# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
qatars_population = 2930524
# From Qatars vaccination data enter populations
full_dose = 2751485 # https://coronavirus.jhu.edu/region/qatar
effective = 1870034 # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
unvaccinated = qatars_population - full_dose
waned = full_dose - effective



#%%
# Seeding infections
qatar_cases_past_day = 790 # https://coronavirus.jhu.edu/region/qatar
total_infections_qatar = qatar_cases_past_day*guess_at_case_to_infection_ratio
sub_populations = [unvaccinated, effective, waned]
for sub_population in sub_populations:
    propotion_of_population = sub_population/qatars_population
    infections_to_seed = (propotion_of_population*total_infections_qatar)
    seeds = world_cup_seeder.calculate_weighting(infection_branch_proportions,
                                                 test_params)





