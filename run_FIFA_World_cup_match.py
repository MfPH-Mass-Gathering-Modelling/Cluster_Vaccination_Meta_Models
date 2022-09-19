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
import math


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

# for now assume England vs Wales Ahmed Bin Ali stadium Tuesday 29th Nov
# Team a: England, Team B: Wales
guess_at_infection_to_detected_ratio = 20 # guess at ratio of actual infections to detected cases ratio
proportion_tickets_to_hosts = 0.1 # proportion of tickets going to host. Informs host spectator population.
stadium_capacity = 40000 # https://hukoomi.gov.qa/en/article/world-cup-stadiums
host_tickets = round(stadium_capacity*proportion_tickets_to_hosts)
visitor_tickets = stadium_capacity - host_tickets
# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
qatars_population = 2930524
# From Qatars vaccination data enter populations
qatar_full_dose = 2751485 # https://coronavirus.jhu.edu/region/qatar
qatar_booster_dose = 1870034 # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
qatar_cases_past_day = 733.429 # 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data

UK_total_vaccinated = 50745901 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
UK_booster_doses = 40373987 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
team_A_proportion_effective = UK_booster_doses/UK_total_vaccinated
team_B_proportion_effective = team_A_proportion_effective
team_A_cases_per_million = 65.368  # UK as whole 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
team_B_cases_per_million  = team_A_cases_per_million
increase_in_transmission_match_day = 2


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
    'p_d': 1 / guess_at_infection_to_detected_ratio,
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
test_params = {
    # Test parameter values - Vaccination parameters
    # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
    # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
    'alpha': 0, # considering the timeframe of this model alpha is not needed.
    'beta_host_spectators': beta,
    'beta_host_spectators_LFDpositive':beta_for_test_positive,
    'beta_host_spectators_PCRpositive':beta_for_test_positive,
    'beta_host_spectators_PCRwaiting': beta,
    'beta_hosts': beta,
    'beta_hosts_LFDpositive': beta_for_test_positive,
    'beta_hosts_PCRpositive': beta_for_test_positive,
    'beta_hosts_PCRwaiting': beta,
    'beta_team_A_supporters': 0, # starts at 0 but will be beta
    'beta_team_A_supporters_LFDpositive': beta_for_test_positive,
    'beta_team_A_supporters_PCRpositive': beta_for_test_positive,
    'beta_team_A_supporters_PCRwaiting': beta,
    'beta_team_B_supporters': 0, # starts at 0 but will be beta
    'beta_team_B_supporters_LFDpositive': beta_for_test_positive,
    'beta_team_B_supporters_PCRpositive': beta_for_test_positive,
    'beta_team_B_supporters_PCRwaiting': beta,
    'epsilon_1': 0.5988023952,
    'epsilon_2': 0.4291845494,
    'epsilon_3': params_for_deriving_beta['epsilon_3'],
    'gamma_A_1': params_for_deriving_beta['gamma_A_1'],
    'gamma_A_2': params_for_deriving_beta['gamma_A_2'],
    'gamma_I_1': params_for_deriving_beta['gamma_I_1'],
    'gamma_I_2': params_for_deriving_beta['gamma_I_2'],
    'gamma_H': 0.0826446281,
    'h_effective': 0.957,
    'h_unvaccinated': 0,
    'h_waned':0.815,
    'iota': 0,  # model runtime is to short to consider end of isolation.
    'kappa_D': kappa,
    'l_effective':0.7443333333,
    'l_unvaccinated': 0,
    'l_waned': 0.2446666667,
    'nu_1':nu,
    'nu_b':nu,
    'nu_w':nu,
    'omega_G':1, # PCR result delay
    'p_s': params_for_deriving_beta['p_s'],
    'p_d': params_for_deriving_beta['p_d'],
    'p_h': params_for_deriving_beta['p_h'],
    's_effective': 0.7486666667,
    's_unvaccinated': 0,
    's_waned': 0.274,
    'tau_A': 0, # Testing is performed by an event.
    'tau_G': 0, # Testing is performed by an event.
    'theta': params_for_deriving_beta['theta']
}

world_cup_model.parameters = test_params



#%%
# Setting up population

def gen_host_sub_popultion(total_pop, full_dose_pop, booster_pop, host_tickets, infections_to_seed):

    host_sub_population = {'hosts': {'unvaccinated': total_pop - full_dose_pop,
                                     'effective': booster_pop,
                                     'waned': full_dose_pop - booster_pop},
                           'host_spectators':{}}

    for sub_pop in ['unvaccinated','effective','waned']:
        spectator_sub_pop = round((host_sub_population['hosts'][sub_pop]/total_pop) * host_tickets)
        host_sub_population['hosts'][sub_pop] -= spectator_sub_pop
        host_sub_population['host_spectators'][sub_pop] = spectator_sub_pop

    for cluster, vac_group_dict in host_sub_population.items():
        for vac_group, population in vac_group_dict.items():
            infections = round(infections_to_seed*(population/total_pop))
            host_sub_population[cluster][vac_group] ={'infections':infections,'S': population-infections}

    return host_sub_population

host_sub_population = gen_host_sub_popultion(qatars_population, qatar_full_dose, qatar_booster_dose, host_tickets,
                                             qatar_cases_past_day * guess_at_infection_to_detected_ratio)




#%%
# Sorting visitor populations
def gen_visitor_sub_population(supportes_pop,
                               supportes_vaccine_prop,
                               detections_per_something,
                               something_denominator,
                               infections_per_detection):
    infections_per_something =  infections_per_detection*detections_per_something
    proportions_total = sum(supportes_vaccine_prop.values())
    if not math.isclose(1, proportions_total, abs_tol=0.000001):
        raise ValueError('The sum of dictionary values in supportes_vaccine_prop should equal 1, it is equal to ' +
                         str(proportions_total) + '.')

    sub_populations = {}
    for vaccine_group, proportion in supportes_vaccine_prop.items():
        sub_population_total = round(supportes_pop*proportion)
        sub_pop_infections = round(sub_population_total*(infections_per_something/something_denominator))
        sub_populations[vaccine_group] = {'S':sub_population_total-sub_pop_infections,
                                          'infections': sub_pop_infections}

    return sub_populations

team_A_vacc_status_proportion = {'unvaccinated': 0,
                                 'effective': team_A_proportion_effective,
                                 'waned': 1-team_A_proportion_effective}
team_B_vacc_status_proportion = {'unvaccinated': 0,
                                 'effective': team_B_proportion_effective,
                                 'waned': 1-team_B_proportion_effective}
tickets_per_team = round(0.5*visitor_tickets)
visitor_sub_pop = {'team_A_supporters': gen_visitor_sub_population(tickets_per_team,
                                                                   team_A_vacc_status_proportion,
                                                                   team_A_cases_per_million,
                                                                   1e6,
                                                                   guess_at_infection_to_detected_ratio),
                   'team_B_supporters': gen_visitor_sub_population(tickets_per_team,
                                                                   team_B_vacc_status_proportion,
                                                                   team_B_cases_per_million,
                                                                   1e6,
                                                                   guess_at_infection_to_detected_ratio)
                   }


#%%
# Setting up infection seeder.
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


#%%
# Seeding infections
all_sub_pops = {**host_sub_population,**visitor_sub_pop}
y0 = np.zeros(world_cup_model.num_state)
state_index = world_cup_model.state_index
for cluster, vaccine_group_dict in all_sub_pops.items():
    for vaccine_group, s_vs_infections in vaccine_group_dict.items():
        state_index_sub_pop = state_index[cluster][vaccine_group]
        y0[state_index_sub_pop['S']] = s_vs_infections['S']

        psv = test_params['p_s']* (1 - test_params['s_'+vaccine_group])
        phv = test_params['p_h']* (1 - test_params['h_'+vaccine_group])
        pd = test_params['p_d']
        asymptomatic_prob = 1 - psv
        symptomatic_prob = psv * (1 - pd)
        detected_prob = psv * pd * (1 - phv)
        hospitalised_prob = psv * pd * phv
        infection_branch_proportions = {'asymptomatic': asymptomatic_prob,
                                        'symptomatic': symptomatic_prob,
                                        'detected': detected_prob,
                                        'hospitalised': hospitalised_prob}
        seeds = world_cup_seeder.seed_infections(s_vs_infections['infections'],
                                                 infection_branch_proportions,
                                                 test_params)
        for state, population in seeds.items():
            y0[state_index_sub_pop[state]] = population



#%%
# Setting up transfer infomation for people being tested
# need to know model end time
end_time = 50
lfd_sensitivity = 0.78 # https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06528-3#:~:text=This%20systematic%20review%20has%20identified,one%20single%20centred%20trial%20(BD
lfd_times = -0.5
lfd_transfer_info = world_cup_model.group_transition_params_dict['tau_A']
lfd_from_index = []
lfd_to_index = []
for vaccine_group_transfer_info in lfd_transfer_info:
    # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
    if vaccine_group_transfer_info['from_cluster'] in ['team_A_supporters','team_B_supporters']:
        lfd_from_index += vaccine_group_transfer_info['from_index']
transfer_info_dict = {'Pre-travel LFD test': {'proportion': lfd_sensitivity,
                                              'from_index': lfd_from_index,
                                              #'to_index': lfd_to_index,
                                              'times': lfd_times,
                                              'type': 'transfer'}}
rtpcr_sensitivity = 0.96 # https://pubmed.ncbi.nlm.nih.gov/34856308/
rtpcr_transfer_info = world_cup_model.group_transition_params_dict['tau_G']
rtpcr_times = -1.5
rtpcr_from_index = []
rtpcr_to_index = []
for vaccine_group_transfer_info in rtpcr_transfer_info:
    # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
    if vaccine_group_transfer_info['from_cluster'] in ['team_A_supporters','team_B_supporters']:
        rtpcr_from_index += vaccine_group_transfer_info['from_index']
transfer_info_dict['Pre-travel RTPCR test'] = {'proportion': rtpcr_sensitivity,
                                               'from_index': rtpcr_from_index,
                                               #'to_index': rtpcr_to_index,
                                               'times': rtpcr_times,
                                               'type': 'transfer'}
transfer_info_dict['supporters arrival'] = {'type': 'change parameter',
                                            'parameters': ['beta_team_A_supporters','beta_team_B_supporters'],
                                            'parameters_getter_method':world_cup_model.parameters,
                                            'parameters_setter_method':world_cup_model.parameters,
                                            'value': beta,
                                            'times': 0}
transfer_info_dict['match day begins'] = {'type': 'change parameter',
                                          'parameters': ['beta_team_A_supporters',
                                                         'beta_team_B_supporters',
                                                         'beta_host_spectators'],
                                          'parameters_getter_method':world_cup_model.parameters,
                                          'parameters_setter_method':world_cup_model.parameters,
                                          'value': beta * increase_in_transmission_match_day,
                                          'times':3}
transfer_info_dict['match day ends'] = {'type': 'change parameter',
                                        'parameters': ['beta_team_A_supporters',
                                                       'beta_team_B_supporters',
                                                       'beta_host_spectators'],
                                        'parameters_getter_method':world_cup_model.parameters,
                                        'parameters_setter_method':world_cup_model.parameters,
                                        'value': beta * increase_in_transmission_match_day,
                                        'times': 4}
transmision_terms = [param for param in test_params.keys() if param.startswith('beta')]
transfer_info_dict['event ends'] = {'type': 'change parameter',
                                        'parameters': transmision_terms,
                                        'parameters_getter_method':world_cup_model.parameters,
                                        'parameters_setter_method':world_cup_model.parameters,
                                        'value': 0,
                                        'times': 7}


transfers_scafold = TranfersAtTsScafold(transfer_info_dict)

#%%
# Runninng mg_model
solution, transfers_df = transfers_scafold.run_simulation(world_cup_model.integrate, y0, end_time,
                                                          start_time=-2, simulation_step=0.5)

#%%
# Checking mg_model is working as it should.


sol_df = results_array_to_df(solution, world_cup_model.state_index)

