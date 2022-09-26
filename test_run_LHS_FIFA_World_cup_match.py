"""
Creation:
    Author: Martin Grunnill
    Date: 14/09/2022
Description: Test run of world cup match simulation with latin hypercube sampling (LHS) and alternate travel screenings.
    
"""
#%%
# import packages

import numpy as np
import json
import os.path

from datetime import datetime
import pandas as pd
# import Quasi-Monte Carlo submodule
from scipy.stats import qmc
from tqdm import tqdm
import pingouin as pg

save_dir = 'LHS test run data'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y %H-%M-%S")

#%%
# LH sampling
LHS_obj = qmc.LatinHypercube(5)
LHS_samples = LHS_obj.random(n=250)
lower_bounds = [1, 1, 1, 0, 0]
upper_bounds = [7, 10, 40, 1, 1]
params_sampled = ['R0', 'increase_in_transmission', 'infection_to_detected', 'isolation', 'LFD_test', ]
samples_df = pd.DataFrame(qmc.scale(LHS_samples, lower_bounds, upper_bounds),
                          columns=params_sampled)
# last out come is binary so:
samples_df.LFD_test = samples_df.LFD_test >= 0.5
samples_df.to_csv(save_dir+'/LH samples ' + dt_string + '.csv', index=False)



# %%
# import model
from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel
# import function for calculating beta from R_0
from CVM_models.Reproductive_Number.Beta_and_Reproduction import MGE_beta_no_vaccine_1_cluster
# import seeding methods
from seeding.multinomail_seeding import MultnomialSeeder
# import transfer class
from event_handling.event_que import EventQueue
from cleaning_up_results.results_to_dfs import results_array_to_df, results_df_pivoted
from setting_up_utils.pop_setup import gen_host_sub_popultion, gen_visitor_sub_population
from setting_up_utils.cluster_params import (list_to_and_from_cluster_param, list_cluster_param,
                                             update_params_with_to_from_cluster_param, update_params_with_cluster_param)

# for now assume England vs Wales Ahmed Bin Ali stadium Tuesday 29th Nov
# Team a: England, Team B: Wales

proportion_tickets_to_hosts = 0.1 # proportion of tickets going to host. Informs host spectator population.
stadium_capacity = 40000 # https://hukoomi.gov.qa/en/article/world-cup-stadiums
host_tickets = round(stadium_capacity*proportion_tickets_to_hosts)
visitor_tickets = stadium_capacity - host_tickets
# Qatar's population 2021
# https://data.worldbank.org/indicator/SP.POP.TOTL?locations=QA
qatars_population = 2930524
host_and_visitor_population = qatars_population +  visitor_tickets
# From Qatars vaccination data enter populations
qatar_full_dose = 2751485 # https://coronavirus.jhu.edu/region/qatar
qatar_booster_dose = 1870034 # assume all booster doses are effectively vaccinated https://covid19.moph.gov.qa/EN/Pages/Vaccination-Program-Data.aspx
qatar_cases_past_day = 733.429 # 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
UK_total_vaccinated = 50745901 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
UK_booster_doses = 40373987 # 2022-09-04 https://github.com/owid/covid-19-data/tree/master/public/data
team_A_proportion_effective = UK_booster_doses/UK_total_vaccinated
team_B_proportion_effective = team_A_proportion_effective
team_A_cases_per_million = 65.368  # UK as whole 7 day smoothed 15/09/2022 https://github.com/owid/covid-19-data/tree/master/public/data
team_B_cases_per_million = team_A_cases_per_million



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

#%%
# set up params not being sampled
nu = 0 # no vaccinations in event
params_not_sampled = {
    # Test parameter values - Vaccination parameters
    # These are available at https://docs.google.com/spreadsheets/d/1_XQn8bfPXKA8r1D_rz6Y1LeMTR_Y0RK5Rh4vnxDcQSo/edit?usp=sharing
    # These are placed in a list corresponding to the order of vaccination groups (enter MassGatheringModel.vaccination_groups)
    'alpha': 0, # considering the timeframe of this model alpha is not needed.
    'epsilon_1': 0.5988023952,
    'epsilon_2': 0.4291845494,
    'epsilon_3': 1,
    'epsilon_H': 0.1627*2,
    'gamma_A_1': 0.139*2,
    'gamma_A_2': 0.139*2,
    'gamma_I_1': 0.1627*2,
    'gamma_I_2': 0.1627*2,
    'gamma_H': 0.0826446281,
    'h_effective': 0.957,
    'h_unvaccinated': 0,
    'h_waned':0.815,
    'iota': 0,  # model runtime is to short to consider end of isolation.
    'l_effective':0.7443333333,
    'l_unvaccinated': 0,
    'l_waned': 0.2446666667,
    'nu_1':nu,
    'nu_b':nu,
    'nu_w':nu,
    'omega_G':1, # PCR result delay
    'p_s': 0.724,
    'p_h': 0.10,
    's_effective': 0.7486666667,
    's_unvaccinated': 0,
    's_waned': 0.274,
    'tau_A': 0, # Testing is performed by an event.
    'tau_G': 0, # Testing is performed by an event.
    'theta': 0.342,
}

#%% Setting up infection seeder.
infection_branches = {'asymptomatic': {'E': 'epsilon_1',
                                       'G_A': 'epsilon_2',
                                       'P_A': 'epsilon_3',
                                       'M_A': 'gamma_A_1',
                                       'F_A': 'gamma_A_2'},
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
                                       'M_H': 'epsilon_H',
                                       'F_H': 'gamma_H'}}

# setting up seeding class
world_cup_seeder = MultnomialSeeder(infection_branches)

#%%


# Setting up event_queue.
start_time = -2
end_time = 50
time_step = 0.5
lfd_sensitivity = 0.78 # https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06528-3#:~:text=This%20systematic%20review%20has%20identified,one%20single%20centred%20trial%20(BD
lfd_transfer_info = world_cup_model.group_transition_params_dict['tau_A']
lfd_from_index = []
lfd_to_index = []
for vaccine_group_transfer_info in lfd_transfer_info:
    # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
    if vaccine_group_transfer_info['from_cluster'] in ['team_A_supporters','team_B_supporters']:
        lfd_from_index += vaccine_group_transfer_info['from_index']
event_info_dict = {'Pre-travel LFD test': {'proportion': lfd_sensitivity,
                                           'from_index': lfd_from_index,
                                           #'to_index': lfd_to_index,
                                           'times': -0.5,
                                           'type': 'transfer'}}
rtpcr_sensitivity = 0.96 # https://pubmed.ncbi.nlm.nih.gov/34856308/
rtpcr_transfer_info = world_cup_model.group_transition_params_dict['tau_G']
rtpcr_from_index = []
rtpcr_to_index = []
for vaccine_group_transfer_info in rtpcr_transfer_info:
    # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
    if vaccine_group_transfer_info['from_cluster'] in ['team_A_supporters','team_B_supporters']:
        rtpcr_from_index += vaccine_group_transfer_info['from_index']
event_info_dict['Pre-travel RTPCR test'] = {'proportion': rtpcr_sensitivity,
                                            'from_index': rtpcr_from_index,
                                            #'to_index': rtpcr_to_index,
                                            'times': -1.5,
                                            'type': 'transfer'}

# useful lists of clusters
all_host_not_positive_clusters = ['hosts', 'hosts_PCR_waiting', 'host_spectators_PCR_waiting', 'host_spectators']
all_vistors_not_positive_clusters = ['team_A_supporters', 'team_A_supporters_PCR_waiting',
                                     'team_B_supporters', 'team_B_supporters_PCR_waiting']
match_attendees = all_vistors_not_positive_clusters + ['host_spectators_PCR_waiting', 'host_spectators']
positive_clusters = [cluster for cluster in world_cup_model.clusters if cluster.endswith('positive')]

pop_visitors_arrive = list_to_and_from_cluster_param('N_',
                                                     all_host_not_positive_clusters + all_vistors_not_positive_clusters,
                                                     all_host_not_positive_clusters + all_vistors_not_positive_clusters)
event_info_dict['supporters arrival pop changes'] = {'type': 'change parameter',
                                                     'changing_parameters': pop_visitors_arrive,
                                                     'times': 0,
                                                     'value': host_and_visitor_population}
beta_visitors_arrive = list_to_and_from_cluster_param('beta_',
                                                      all_host_not_positive_clusters + all_vistors_not_positive_clusters,
                                                      all_vistors_not_positive_clusters)
event_info_dict['supporters arrival beta changes'] = {'type': 'change parameter',
                                                      'changing_parameters': beta_visitors_arrive,
                                                      'times': 0}

pop_match_day_begins = list_to_and_from_cluster_param('N_', world_cup_model.clusters, match_attendees)
attendee_index = world_cup_model.get_clusters_indexes(match_attendees)
event_info_dict['match day begins pop changes attendes'] = {'type': 'parameter equals subpopulation',
                                                            'changing_parameters': pop_match_day_begins,
                                                            'times':3,
                                                            'subpopulation_index': attendee_index}

not_attending = ['hosts', 'hosts_PCR_waiting'] + positive_clusters
pop_match_day_begins_non_attendees = list_to_and_from_cluster_param('N_', world_cup_model.clusters, not_attending)
non_attendee_index = world_cup_model.get_clusters_indexes(not_attending)
event_info_dict['match day begins pop changes non-attendes'] = {'type': 'parameter equals subpopulation',
                                                                'changing_parameters': ['hosts', 'hosts_PCR_waiting'],
                                                                'times':3,
                                                                'subpopulation_index': non_attendee_index}
beta_match_day = list_to_and_from_cluster_param('beta_', match_attendees, match_attendees)
event_info_dict['match day begins beta changes'] = {'type': 'change parameter',
                                                    'changing_parameters': beta_match_day,
                                                    'times':3}


event_info_dict['match day ends pop changes'] = {'type': 'change parameter',
                                                 'changing_parameters': pop_visitors_arrive,
                                                 'times': 4,
                                                 'value': host_and_visitor_population}
event_info_dict['match day ends beta'] = {'type': 'change parameter',
                                          'changing_parameters': beta_match_day,
                                          'times': 4}
transmision_terms = [param for param in world_cup_model.all_parameters if param.startswith('beta')]
event_info_dict['MGE ends'] = {'type': 'change parameter',
                               'changing_parameters': transmision_terms,
                               'value': 0,
                               'times': 7}


match_event_queue = EventQueue(event_info_dict)

time = pd.Series(np.arange(start_time, end_time+time_step, time_step))
time.to_csv(save_dir+'/simulation time steps.csv', index=False)

#%%

sims_dfs = []
tranfers_dfs = []
# helpful dictionaries for sorting out transmission.
to_tranmission_terms = world_cup_model.transmission_to_terms
from_tranmission_terms = world_cup_model.transmission_from_terms

#%% Useful functions



# itterating over sample Dataframe rows
# for sample_index, sample in tqdm(samples_df.iterrows(),
#                                  desc='Latin Hypercube Sample', total=len(samples_df)):
sample_index , sample = next(samples_df.iterrows())
    kappa = sample['isolation']
    p_d = 0.1
    params_needed_for_beta_not_sampled = ['epsilon_3',
                                          'p_s',
                                          'p_h',
                                          'gamma_A_1',
                                          'gamma_A_2',
                                          'gamma_I_1',
                                          'gamma_I_2',
                                          'theta']
    params_for_deriving_beta = {param: params_not_sampled[param]
                                for param in params_needed_for_beta_not_sampled}
    params_for_deriving_beta['kappa_D'] = kappa
    params_for_deriving_beta['p_d'] = p_d
    beta = MGE_beta_no_vaccine_1_cluster(R_0=sample['R0'], **params_for_deriving_beta)
    # beta between host clusters starts at default

    test_params = {}
    update_params_with_to_from_cluster_param(test_params,
                                             all_host_not_positive_clusters, all_host_not_positive_clusters,
                                             'beta', beta)
    params_to_change = update_params_with_to_from_cluster_param(test_params,
                                                                all_host_not_positive_clusters, all_host_not_positive_clusters,
                                                                'N', qatars_population,
                                                                return_list=True) # starts off just Qatari population


    # positive will be isolating but interact with all so ...
    beta_from_test_positive = beta*kappa
    update_params_with_to_from_cluster_param(test_params,
                                             world_cup_model.clusters,
                                             positive_clusters,
                                             'beta',
                                             beta_from_test_positive)
    update_params_with_to_from_cluster_param(test_params,
                                             world_cup_model.clusters,
                                             positive_clusters,
                                             'N',
                                             host_and_visitor_population)
    # positive clusters only include infecteds in this model (they cannot be infected again so):
    update_params_with_to_from_cluster_param(test_params,
                                             positive_clusters,
                                             world_cup_model.clusters,
                                             'beta',
                                             0)
    update_params_with_to_from_cluster_param(test_params,
                                             positive_clusters,
                                             world_cup_model.clusters,
                                             'N',
                                             host_and_visitor_population)

    # %% dealing with visitor transmission

    all_vistors_not_positive_clusters





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
        **params_not_sampled,
        **params_for_deriving_beta
    }

    # Setting up host population
    host_sub_population = gen_host_sub_popultion(qatars_population, qatar_full_dose, qatar_booster_dose, host_tickets,
                                                 qatar_cases_past_day * sample['infection_to_detected'])

    # Sorting visitor populations
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
                                                                       sample['infection_to_detected']),
                       'team_B_supporters': gen_visitor_sub_population(tickets_per_team,
                                                                       team_B_vacc_status_proportion,
                                                                       team_B_cases_per_million,
                                                                       1e6,
                                                                       sample['infection_to_detected'])
                       }

    # Seeding infections
    all_sub_pops = {**host_sub_population,**visitor_sub_pop}
    y0 = np.zeros(world_cup_model.num_state)
    state_index = world_cup_model.state_index
    for cluster, vaccine_group_dict in all_sub_pops.items():
        for vaccine_group, s_vs_infections in vaccine_group_dict.items():
            state_index_sub_pop = state_index[cluster][vaccine_group]
            y0[state_index_sub_pop['S']] = s_vs_infections['S']

            p_s_v = test_params['p_s'] * (1 - test_params['s_' + vaccine_group])
            p_h_v = test_params['p_h'] * (1 - test_params['h_' + vaccine_group])
            asymptomatic_prob = 1 - p_s_v
            symptomatic_prob = p_s_v * (1 - p_d)
            detected_prob = p_s_v * p_d * (1 - p_h_v)
            hospitalised_prob = p_s_v * p_d * p_h_v
            infection_branch_proportions = {'asymptomatic': asymptomatic_prob,
                                            'symptomatic': symptomatic_prob,
                                            'detected': detected_prob,
                                            'hospitalised': hospitalised_prob}
            seeds = world_cup_seeder.seed_infections(s_vs_infections['infections'],
                                                     infection_branch_proportions,
                                                     test_params)
            for state, population in seeds.items():
                y0[state_index_sub_pop[state]] = population


    # change event values and proportions
    match_event_queue.change_event_value('supporters arrival', beta)
    match_event_queue.change_event_factor('match day begins', sample['increase_in_transmission'])
    match_event_queue.change_event_value('match day ends', beta)
    if sample['LFD_test']:
        match_event_queue.change_event_proportion('Pre-travel LFD test', lfd_sensitivity)
        match_event_queue.make_events_nullevents('Pre-travel RTPCR test')
    else:
        match_event_queue.change_event_proportion('Pre-travel RTPCR test', rtpcr_sensitivity)
        match_event_queue.make_events_nullevents('Pre-travel LFD test')




    # Runninng mg_model
    world_cup_model.parameters = test_params
    solution, transfers_df = match_event_queue.run_simulation(model_object=world_cup_model,
                                                              run_attribute='integrate',
                                                              parameters=test_params, parameters_attribute='parameters',
                                                              y0=y0,
                                                              end_time=end_time, start_time=start_time, simulation_step=time_step)
    sol_df = results_array_to_df(solution, world_cup_model.state_index,
                                 start_time=start_time, simulation_step=time_step, end_time=end_time)
    save_prefix = save_dir +'/LHS ' + str(sample_index)
    # sol_df.to_csv(save_prefix+' simulation.csv', index=False)
    # transfers_df.to_csv(save_prefix+' tranfers.csv', index=False)
    sims_dfs.append(sol_df)
    tranfers_dfs.append(transfers_df)



#%% Caluculating total hostpitalisaions
hospitalisations = []
for sims_df in sims_dfs:
    hospitalisations.append(sims_df.iloc[-1,-1])

samples_df["hospitalisations"] = hospitalisations
rank_pcors = []
for index, param in enumerate(params_sampled):
    covar = [item
             for item in params_sampled
             if item not in [param, 'hospitalisations']]
    param_rank_pcor = pg.partial_corr(samples_df,
                                      x=param, y='hospitalisations',
                                      covar= covar,
                                      method='spearman')
    rank_pcors.append(param_rank_pcor)

rank_pcors = pd.concat(rank_pcors, ignore_index=True)
rank_pcors.index = params_sampled

rank_pcors.to_csv(save_dir+'/PRCors Hospitalisations ' + dt_string + '.csv')