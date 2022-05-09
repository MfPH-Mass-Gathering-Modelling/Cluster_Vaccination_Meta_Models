"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import copy
from pygom import TransitionType, Transition
from CVM_models.pygom_models.base_vaccination import BaseMultiClusterVacConstructor


class MGModelConstructor(BaseMultiClusterVacConstructor):
    states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2', 'H', 'D', 'R']
    dead_states = ['D']
    vaccinable_states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'R']
    infectious_states = ['T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2']
    symptomatic_states = ['I_1', 'I_2', 'H']
    non_specific_params = ['theta', 'epsilon_1', 'epsilon_2', 'epsilon_3',
                           'gamma_i_1', 'gamma_i_2', 'gamma_a_1', 'gamma_a_2',
                           'psi', 'rho', 'alpha']
    cluster_specific_params = BaseMultiClusterVacConstructor.cluster_specific_params + ['eta', 'mu']
    vaccine_specific_params = ['l', 'r', 'h', 'm']

    def __init__(self, clusters, vaccine_groups, include_observed_states=True):
        if include_observed_states:
            self.observed_states = ['H_T', 'D_T']
            total_hospitalised = []
            total_recovered_from_hosptial = []
            total_dead = []
        super().__init__(clusters, vaccine_groups)
        for vaccine_stage, vaccine_group in enumerate(self.vaccine_groups):
            l_v = 'l_' + vaccine_group
            r_v = 'r_' + vaccine_group
            h_v = 'h_' + vaccine_group
            m_v = 'm_' + vaccine_group
            for cluster_i in self.clusters:
                eta_i = 'eta_' + cluster_i
                mu_i = 'mu_' + cluster_i
                lambda_i = 'lambda_' + cluster_i

                S_i_v = "S_" + cluster_i + '_' + vaccine_group
                E_i_v = "E_" + cluster_i + '_' + vaccine_group
                T_i_v = "T_" + cluster_i + '_' + vaccine_group
                T_I_i_v = "T_I_" + cluster_i + '_' + vaccine_group
                T_A_i_v = "T_A_" + cluster_i + '_' + vaccine_group
                A_1_i_v = "A_1_" + cluster_i + '_' + vaccine_group
                A_2_i_v = "A_2_" + cluster_i + '_' + vaccine_group
                I_1_i_v = "I_1_" + cluster_i + '_' + vaccine_group
                I_2_i_v = "I_2_" + cluster_i + '_' + vaccine_group
                D_i_v = "D_" + cluster_i + '_' + vaccine_group
                H_i_v = "H_" + cluster_i + '_' + vaccine_group
                R_i_v = "R_" + cluster_i + '_' + vaccine_group

                hospitalised = eta_i + '*(1-' + h_v + ')*' + I_2_i_v
                hospital_recovery = 'psi*' + H_i_v
                symptomatic_deaths = mu_i + '*(1-' + m_v + ')*' + I_2_i_v
                hospital_deaths = mu_i + '*(1-' + m_v + ')*' + H_i_v
                if include_observed_states:
                    total_hospitalised.append(hospitalised)
                    total_recovered_from_hosptial.append(hospital_recovery)
                    total_dead.append(symptomatic_deaths + '+' + hospital_deaths)

                self.transitions += [
                    Transition(origin=S_i_v, destination=E_i_v,
                               equation=lambda_i + '*' + '(1-' + l_v + ')*' + S_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=E_i_v, destination=T_i_v,
                               equation='epsilon_1*' + E_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=T_i_v, destination=T_I_i_v,
                               equation='epsilon_2 * rho *(1-' + r_v + ') *' + T_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=T_i_v, destination=T_A_i_v,
                               equation='epsilon_2 *(1- rho *(1-' + r_v + ')) *' + T_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=T_A_i_v, destination=A_1_i_v,
                               equation='epsilon_3 *' + T_A_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=A_1_i_v, destination=A_2_i_v,
                               equation='gamma_a_1*' + A_1_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=A_2_i_v, destination=R_i_v,
                               equation='gamma_a_2*' + A_2_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=T_I_i_v, destination=I_1_i_v,
                               equation='epsilon_3 *' + T_I_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=I_1_i_v, destination=I_2_i_v,
                               equation='gamma_i_1*' + I_1_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=I_2_i_v, destination=R_i_v,
                               equation='gamma_i_2*' + I_2_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=I_2_i_v, destination=H_i_v,
                               equation=hospitalised,
                               transition_type=TransitionType.T),
                    Transition(origin=H_i_v, destination=R_i_v,
                               equation=hospital_recovery,
                               transition_type=TransitionType.T),
                    Transition(origin=I_2_i_v, destination=D_i_v,
                               equation=symptomatic_deaths,
                               transition_type=TransitionType.T),
                    Transition(origin=H_i_v, destination=D_i_v,
                               equation=hospital_deaths,
                               transition_type=TransitionType.T),
                    Transition(origin=R_i_v, destination=S_i_v,
                               equation='alpha*'+ R_i_v,
                               transition_type=TransitionType.T)
                ]

        if include_observed_states:
            self.bd_list += [
                Transition(origin='H_T',
                           equation=' + '.join(total_hospitalised),
                           transition_type=TransitionType.B),
                Transition(origin='H_T',
                           equation=' + '.join(total_recovered_from_hosptial),
                           transition_type=TransitionType.D),
                Transition(origin='D_T',
                           equation=' + '.join(total_dead),
                           transition_type=TransitionType.B)
            ]