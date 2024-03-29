"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
from CVM_models.pure_python_models.base_piecewise_vaccination import BaseScipyClusterVacModel
import os

# find this files directory so as to later find jacobians saved a json files
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath) +'/'


class MassGatheringModel(BaseScipyClusterVacModel):
    states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_H', 'M_I', 'M_A', 'F_H', 'F_I', 'F_A', 'R']
    observed_states = ['Cumulative hospitalisation', 'Cumulative infections']
    infected_states = ['E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_H',  'M_I', 'M_A', 'F_H',  'F_I', 'F_A']
    hospitalised_states = ['F_H']
    infectious_states = ['P_I', 'P_A',  'M_I', 'M_A', 'M_H',  'F_I', 'F_A']
    symptomatic_states = ['M_I', 'F_I',  'M_H', 'F_H']
    non_transmission_universal_params = ['epsilon_1', 'epsilon_2', 'epsilon_3', 'epsilon_H',
                                         'p_s', 'p_h_s',
                                         'gamma_A_1', 'gamma_A_2',
                                         'gamma_I_1', 'gamma_I_2',
                                         'gamma_H',
                                         'alpha'
                                         ]
    vaccine_specific_params = ['l','s','h']
    transmission_cluster_specfic = True
    isolation_cluster_specfic = True
    asymptomatic_transmission_modifier = 'theta'


    def ode(self, y, t, *parameters):
        parameters = self._sorting_params(parameters)
        y_deltas = np.zeros(self.num_state)
        fois = self.foi(y, parameters)
        for cluster in self.clusters:
            foi = fois[cluster]
            for vaccine_group in self.vaccine_groups:
                ve_infection = 'l_'+vaccine_group
                ve_symptoms = 's_' + vaccine_group
                ve_hospitalisation = 'h_' + vaccine_group
                self.group_transfer(y, y_deltas, t, cluster, vaccine_group, parameters)
                states_index = self.state_index[cluster][vaccine_group]
                # Infections
                infections = (1 - parameters[ve_infection]) * foi * y[states_index['S']]
                # progression to RT-pcr sensitivity
                prog_rtpcr = parameters['epsilon_1'] * y[states_index['E']]
                p_s_v = parameters['p_s'] * (1 - parameters[ve_symptoms])
                prog_symptomatic_path = prog_rtpcr * p_s_v
                prog_asymptomatic_path = prog_rtpcr * (1-p_s_v)


                # progression to lfd/rapid antigen test sensitivity
                prog_LFD_sensitive_symptomatic_path = parameters['epsilon_2'] * y[states_index['G_I']]
                prog_LFD_sensitive_asymptomatic_path = parameters['epsilon_2'] * y[states_index['G_A']]


                # progression to mid-infection
                prog_symptoms = parameters['epsilon_3'] * y[states_index['P_I']]
                p_h_v = parameters['p_h_s'] * (1 - parameters[ve_hospitalisation])
                prog_hospital_path = p_h_v*prog_symptoms
                prog_not_hospital_path = prog_symptoms-prog_hospital_path

                prog_mid_asymptomatic_stage = parameters['epsilon_3'] * y[states_index['P_A']]

                #Late infection
                prog_late_asymptomatic_stage = parameters['gamma_A_1'] * y[states_index['M_A']]
                prog_late_symptomatic_stage = parameters['gamma_I_1'] * y[states_index['M_I']]
                hospitalisation = parameters['epsilon_H'] * y[states_index['M_H']]

                # recovery
                asymptomatic_recovery = parameters['gamma_A_2'] * y[states_index['F_A']]
                symptomatic_recovery = parameters['gamma_I_2'] * y[states_index['F_I']]
                hospital_recovery = parameters['gamma_H'] * y[states_index['F_H']]

                # natural wanning imunity
                waned_natural_immunity = parameters['alpha'] * y[states_index['R']]

                # Put together deltas
                y_deltas[states_index['S']] += waned_natural_immunity - infections
                y_deltas[states_index['E']] += infections - prog_rtpcr
                y_deltas[states_index['G_I']] += prog_symptomatic_path - prog_LFD_sensitive_symptomatic_path
                y_deltas[states_index['G_A']] += prog_asymptomatic_path - prog_LFD_sensitive_asymptomatic_path
                y_deltas[states_index['P_I']] += prog_LFD_sensitive_symptomatic_path - prog_symptoms
                y_deltas[states_index['P_A']] += prog_LFD_sensitive_asymptomatic_path - prog_mid_asymptomatic_stage
                y_deltas[states_index['M_A']] += prog_mid_asymptomatic_stage-prog_late_asymptomatic_stage
                y_deltas[states_index['M_I']] += prog_not_hospital_path-prog_late_symptomatic_stage
                y_deltas[states_index['M_H']] += prog_hospital_path - hospitalisation
                y_deltas[states_index['F_A']] += prog_late_asymptomatic_stage-asymptomatic_recovery
                y_deltas[states_index['F_I']] += prog_late_symptomatic_stage-symptomatic_recovery
                y_deltas[states_index['F_H']] += hospitalisation - hospital_recovery
                y_deltas[states_index['R']] += hospital_recovery + asymptomatic_recovery + symptomatic_recovery - waned_natural_immunity
                y_deltas[-2] += hospitalisation
                y_deltas[-1] += infections
                # self.ode_calls_dict[key] = y_deltas

        return y_deltas
