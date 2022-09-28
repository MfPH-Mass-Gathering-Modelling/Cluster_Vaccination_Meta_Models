"""
Creation:
    Author: Martin Grunnill
    Date: 2022-09-27
Description: Set up an object for running world cup match MGE modelling simulations.
    
"""
import json
import numpy as np
import inspect

from CVM_models.pure_python_models.mass_gathering_piecewise_vaccination import MassGatheringModel
# import function for calculating beta from R_0
from CVM_models.Reproductive_Number.Beta_and_Reproduction import MGE_beta_no_vaccine_1_cluster
# import seeding methods
from seeding.multinomail_seeding import MultnomialSeeder
from event_handling.event_que import EventQueue
from cleaning_up_results.results_to_dfs import results_array_to_df, results_df_pivoted
from setting_up_utils.pop_setup import gen_host_sub_popultion, gen_visitor_sub_population
from setting_up_utils.cluster_params import (list_to_and_from_cluster_param, list_cluster_param,
                                             update_params_with_to_from_cluster_param, update_params_with_cluster_param)


# Load metapopulation informations
structures_dir = ('C:/Users/mdgru/OneDrive - York University/Documents/York University Postdoc/'+
                  'Mass Gathering work/Compartment based models/Cluster_Vaccination_Meta_Models/'+
                  'CVM_models/Model meta population structures/')
with open(structures_dir + "Sports match MGE.json", "r") as json_file:
    metapoplulation_info=json_file.read()

metapoplulation_info = json.loads(metapoplulation_info)


class SportMatchMGESimulation:
    def __init__(self,
                 host_population, host_cases_per_million,
                 total_hosts_vaccinated, hosts_effectively_vaccinated,
                 stadium_capacity,
                 team_A_cases_per_million, team_B_cases_per_million,
                 fixed_parameters={}):
        self.host_population = host_population
        self.host_cases_per_million = host_cases_per_million
        self.hosts_unvaccinated = host_population - total_hosts_vaccinated
        self.hosts_waned_vaccinated = total_hosts_vaccinated - hosts_effectively_vaccinated
        self.hosts_effectively_vaccinated = hosts_effectively_vaccinated
        self.stadium_capacity = stadium_capacity
        self.team_A_cases_per_million = team_A_cases_per_million
        self.team_B_cases_per_million = team_B_cases_per_million
        self.model = MassGatheringModel(metapoplulation_info)
        self.fixed_parameters = fixed_parameters
        self._setup_seeder()
        self._setup_event_queue()


    def _setup_seeder(self):
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
        self.seeder = MultnomialSeeder(infection_branches)


    def _setup_event_queue(self):
        # useful lists of clusters
        self.hosts_main = ['hosts', 'host_spectators','host_staff']
        self.host_not_positive = (self.hosts_main +
                                  [cluster +'_PCR_waiting' for cluster in self.hosts_main])
        self.vistors_main = ['team_A_supporters', 'team_B_supporters']
        self.vistors_not_positive = (self.vistors_main +
                                     [cluster +'_PCR_waiting' for cluster in self.vistors_main])
        self.positive_clusters = [cluster for cluster in self.model.clusters if cluster.endswith('positive')]


        self.match_attendees_main = self.vistors_main + ['host_spectators', 'host_staff']
        self.match_attendees_not_positive = (self.match_attendees_main +
                                             [cluster +'_PCR_waiting' for cluster in self.match_attendees_main])
        self.not_attending_match = ['hosts', 'hosts_PCR_waiting'] + self.positive_clusters


        # Setting up event_queue.
        self.start_time = -2
        self.end_time = 75
        self.time_step = 0.5 # https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-021-06528-3#:~:text=This%20systematic%20review%20has%20identified,one%20single%20centred%20trial%20(BD

        event_info_dict = {}

        # Setup Tests events
        lfd_transfer_info = self.model.group_transition_params_dict['tau_A']
        lfd_from_index = set()
        lfd_to_index = set()
        for vaccine_group_transfer_info in lfd_transfer_info:
            # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
            if vaccine_group_transfer_info['from_cluster'] in self.vistors_main:
                lfd_from_index.update(vaccine_group_transfer_info['from_index'])
        event_info_dict['Pre-travel LFD'] = {'from_index': list(lfd_from_index),
                                             # 'to_index': list(lfd_to_index),
                                             'times': -0.5,
                                             'type': 'transfer'}
        for vaccine_group_transfer_info in lfd_transfer_info:
            if vaccine_group_transfer_info['from_cluster'] in self.match_attendees_main:
                lfd_from_index.update(vaccine_group_transfer_info['from_index'])
                lfd_to_index.update(vaccine_group_transfer_info['to_index'])
        event_info_dict['Pre-match LFD'] = {'from_index': list(lfd_from_index),
                                            'to_index': list(lfd_to_index),
                                            'times': 2.5,
                                            'type': 'transfer'}
        event_info_dict['Post-match LFD'] = {'from_index': list(lfd_from_index),
                                             'to_index': list(lfd_to_index),
                                             'times': 6.5,
                                             'type': 'transfer'}


        rtpcr_transfer_info = self.model.group_transition_params_dict['tau_G']
        rtpcr_from_index = set()
        rtpcr_to_index = set()
        for vaccine_group_transfer_info in rtpcr_transfer_info:
            # for now we are only interested in pre-travel screen so removing from team_a and team_b supporters, not tranfering.
            if vaccine_group_transfer_info['from_cluster'] in self.vistors_main:
                rtpcr_from_index.update(vaccine_group_transfer_info['from_index'])
        event_info_dict['Pre-travel RTPCR'] = {'from_index': list(rtpcr_from_index),
                                               # 'to_index': list(rtpcr_to_index),
                                               'times': -1.5,
                                               'type': 'transfer'}
        for vaccine_group_transfer_info in rtpcr_transfer_info:
            if vaccine_group_transfer_info['from_cluster'] in self.match_attendees_main:
                rtpcr_from_index.update(vaccine_group_transfer_info['from_index'])
                rtpcr_to_index.update(vaccine_group_transfer_info['to_index'])
        event_info_dict['Pre-match RTPCR'] = {'from_index': list(rtpcr_from_index),
                                              'to_index': list(rtpcr_to_index),
                                              'times': 1.5,
                                              'type': 'transfer'}
        event_info_dict['Post-match RTPCR'] = {'from_index': list(rtpcr_from_index),
                                               'to_index': list(rtpcr_to_index),
                                               'times': 5.5,
                                               'type': 'transfer'}

        pop_visitors_arrive = list_to_and_from_cluster_param('N_',
                                                             self.host_not_positive + self.vistors_not_positive,
                                                             self.host_not_positive + self.vistors_not_positive)
        all_clusters_index = self.model.get_clusters_indexes(self.model.clusters)
        event_info_dict['visitor arrival pop changes'] = {'type': 'parameter equals subpopulation',
                                                          'changing_parameters': pop_visitors_arrive,
                                                          'times': 0,
                                                          'subpopulation_index': all_clusters_index}
        beta_visitors_arrive = list_to_and_from_cluster_param('beta_',
                                                              self.host_not_positive + self.vistors_not_positive,
                                                              self.vistors_not_positive)
        event_info_dict['visitor arrival beta changes'] = {'type': 'change parameter',
                                                           'changing_parameters': beta_visitors_arrive,
                                                           'times': 0}

        pop_match_day_begins = list_to_and_from_cluster_param('N_', self.model.clusters, self.match_attendees_not_positive)
        attendee_index = self.model.get_clusters_indexes(self.match_attendees_not_positive)
        event_info_dict['match day begins pop changes attendees'] = {'type': 'parameter equals subpopulation',
                                                                     'changing_parameters': pop_match_day_begins,
                                                                     'times': 3,
                                                                     'subpopulation_index': attendee_index}


        pop_match_day_begins_non_attendees = list_to_and_from_cluster_param('N_', self.model.clusters,
                                                                            self.not_attending_match)
        non_attendee_index = self.model.get_clusters_indexes(self.not_attending_match)
        event_info_dict['match day begins pop changes non-attendees'] = {'type': 'parameter equals subpopulation',
                                                                         'changing_parameters': pop_match_day_begins_non_attendees,
                                                                         'times': 3,
                                                                         'subpopulation_index': non_attendee_index}
        beta_match_day = list_to_and_from_cluster_param('beta_', self.match_attendees_not_positive, self.match_attendees_not_positive)
        event_info_dict['match day begins beta changes'] = {'type': 'change parameter',
                                                            'changing_parameters': beta_match_day,
                                                            'times': 3}

        event_info_dict['match day ends pop changes'] = {'type': 'parameter equals subpopulation',
                                                         'changing_parameters': pop_visitors_arrive,
                                                         'times': 4,
                                                         'subpopulation_index': all_clusters_index}
        event_info_dict['match day ends beta'] = {'type': 'change parameter',
                                                  'changing_parameters': beta_match_day,
                                                  'times': 4}
        transmision_terms = [param for param in self.model.all_parameters if param.startswith('beta')]
        event_info_dict['MGE ends'] = {'type': 'change parameter',
                                       'changing_parameters': transmision_terms,
                                       'value': 0,
                                       'times': 7}
        self.event_queue = EventQueue(event_info_dict)


    def run_simulation(self, sampled_parameters):
        # reset any changes in event queue caused by previously running self.run_simulation
        self.event_queue.reset_event_queue()
        parameters = {**self.fixed_parameters, **sampled_parameters}
        # Setup population
        host_tickets = round(self.stadium_capacity * parameters['proportion host tickets'])
        visitor_tickets = self.stadium_capacity - host_tickets
        host_and_visitor_population = self.host_population + visitor_tickets
        staff = round(self.stadium_capacity*parameters['staff per ticket'])
        
        
        
        # Setup starting transmission and changes in transmission
        beta_derive_func_args = list(inspect.signature(MGE_beta_no_vaccine_1_cluster).parameters.keys())
        params_for_deriving_beta = {param: parameters[param]
                                    for param in beta_derive_func_args}
        beta = MGE_beta_no_vaccine_1_cluster(**params_for_deriving_beta)
        
        # starting transmission terms
        # all positive clusters will be isolating but interact with all so ...
        beta_from_test_positive = beta * parameters['kappa']
        update_params_with_to_from_cluster_param(parameters,
                                                 self.model.clusters,
                                                 self.positive_clusters,
                                                 'beta',
                                                 beta_from_test_positive)
        # There population only gets added to once pre match testing begins so ..
        update_params_with_to_from_cluster_param(parameters,
                                                 self.model.clusters,
                                                 self.positive_clusters,
                                                 'N',
                                                 host_and_visitor_population)
        # positive clusters only include infecteds in this model (they cannot be infected again so):
        update_params_with_to_from_cluster_param(parameters,
                                                 self.positive_clusters,
                                                 self.model.clusters,
                                                 'beta',
                                                 0)
        update_params_with_to_from_cluster_param(parameters,
                                                 self.positive_clusters,
                                                 self.model.clusters,
                                                 'N',
                                                 host_and_visitor_population)
        
        # beta between host clusters starts at default and stays same until 'MGE ends' event
        update_params_with_to_from_cluster_param(parameters,
                                                 self.host_not_positive, self.host_not_positive,
                                                 'beta', beta)
        # all transmission events to hosts happens in there population to begin with.
        update_params_with_to_from_cluster_param(parameters,
                                                 self.host_not_positive, self.model.clusters,
                                                 'N',
                                                 self.host_population)  # starts off just Qatari population
        
        
        
        # all visitor clusters do not transmit to begin with.
        update_params_with_to_from_cluster_param(parameters,
                                                 self.model.clusters, self.vistors_not_positive,
                                                 'beta', 0)
        # all vistor population are half the number of visitor tickets to begin with
        update_params_with_to_from_cluster_param(parameters,
                                                 self.vistors_not_positive, self.model.clusters,
                                                 'N', round(visitor_tickets/2))
        
        # visitors arrive transmission terms changes
        self.event_queue.change_event_value('visitor arrival beta changes', beta)

        # match day begins
        self.event_queue.change_event_value('match day begins beta changes', beta * parameters['increase in transmission'])

        # match day ends
        self.event_queue.change_event_value('match day ends beta changes', beta)

        # change test_event values         
        if parameters['Pre-travel test']:
            self.event_queue.change_event_proportion('Pre-travel ' + parameters['test type'],
                                                     parameters['test sensitivity'])
        if parameters['Pre-match test']:
            self.event_queue.change_event_proportion('Pre-match ' + parameters['test type'],
                                                     parameters['test sensitivity'])
        if parameters['Post-match test']:
            self.event_queue.change_event_proportion('Post-match ' + parameters['test type'],
                                                     parameters['test sensitivity'])

        # setting up populations
        # Setting up host population
        host_sub_population = gen_host_sub_popultion(self.host_population, host_tickets, staff,
                                                     self.hosts_unvaccinated, self.hosts_effectively_vaccinated, self.hosts_waned_vaccinated,
                                                     self.host_cases_per_million, parameters['host infections per case'])

        # Sorting visitor populations
        team_A_vacc_status_proportion = {'unvaccinated': 0,
                                         'effective': parameters['team A prop effectively vaccinated'],
                                         'waned': 1 - parameters['team A prop effectively vaccinated']}
        team_B_vacc_status_proportion = {'unvaccinated': 0,
                                         'effective': parameters['team B prop effectively vaccinated'],
                                         'waned': 1 - parameters['team B prop effectively vaccinated']}
        tickets_per_team = round(0.5 * visitor_tickets)
        visitor_sub_pop = {'team_A_supporters': gen_visitor_sub_population(tickets_per_team,
                                                                           team_A_vacc_status_proportion,
                                                                           parameters['team A cases per million'],
                                                                           parameters['team A infections per case']),
                           'team_B_supporters': gen_visitor_sub_population(tickets_per_team,
                                                                           team_B_vacc_status_proportion,
                                                                           parameters['team B cases per million'],
                                                                           parameters['team B infections per case'])
                           }

        # Seeding infections
        all_sub_pops = {**host_sub_population, **visitor_sub_pop}
        y0 = np.zeros(self.model.num_state)
        state_index = self.model.state_index
        for cluster, vaccine_group_dict in all_sub_pops.items():
            for vaccine_group, s_vs_infections in vaccine_group_dict.items():
                state_index_sub_pop = state_index[cluster][vaccine_group]
                y0[state_index_sub_pop['S']] = s_vs_infections['S']
                p_s_v = parameters['p_s'] * (1 - parameters['s_' + vaccine_group])
                p_h_v = parameters['p_h'] * (1 - parameters['h_' + vaccine_group])
                asymptomatic_prob = 1 - p_s_v
                symptomatic_prob = p_s_v * (1 - parameters['p_d'])
                detected_prob = p_s_v * parameters['p_d'] * (1 - p_h_v)
                hospitalised_prob = p_s_v * parameters['p_d'] * p_h_v
                infection_branch_proportions = {'asymptomatic': asymptomatic_prob,
                                                'symptomatic': symptomatic_prob,
                                                'detected': detected_prob,
                                                'hospitalised': hospitalised_prob}
                seeds = self.seeder.seed_infections(s_vs_infections['infections'],
                                                    infection_branch_proportions,
                                                    parameters)
                for state, population in seeds.items():
                    y0[state_index_sub_pop[state]] = population

        # Runninng mg_model
        parameters_needed_for_model = {parameter for parameter in parameters if parameter in self.model.all_parameters}
        self.model.parameters = parameters_needed_for_model
        solution, transfers_df = self.event_queue.run_simulation(model_object=self.model,
                                                                 run_attribute='integrate',
                                                                 parameters=parameters_needed_for_model,
                                                                 parameters_attribute='parameters',
                                                                 y0=y0,
                                                                 end_time=self.end_time, start_time=self.start_time,
                                                                 simulation_step=self.time_step)
        sol_df = results_array_to_df(solution, self.model.state_index,
                                     start_time=self.start_time, simulation_step=self.time_step, end_time=self.end_time)
        if 'save prefix' in parameters.keys():
            sol_df.to_csv(parameters['save prefix'] + ' simulation.csv', index_label='time')
            transfers_df.to_csv(parameters['save prefix']+' tranfers.csv', index=False)
        else:
            return sol_df, transfers_df





        
