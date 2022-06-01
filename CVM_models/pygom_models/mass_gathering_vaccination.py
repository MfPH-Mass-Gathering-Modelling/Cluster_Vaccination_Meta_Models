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
    states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_D', 'M_I', 'M_A', 'F_D', 'F_I', 'F_A', 'H', 'R']
    vaccinable_states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_A', 'F_A', 'R']
    vaccine_dict ={'unvaccinated': vaccinable_states,
                   'first_dose_delay': states,
                   'first_dose': vaccinable_states,
                   'second_dose_delay': states,
                   'second_dose': states,
                   'waned': vaccinable_states,
                   'third_dose_delay': states,
                   'third_dose':None}
    infectious_states = ['P_I', 'P_A', 'M_D', 'M_I', 'M_A', 'F_D', 'F_I', 'F_A']
    symptomatic_states = ['M_I','F_I']
    isolating_states = ['M_D','F_D']
    non_specific_params = ['theta', 'p_s', 'epsilon_1', 'epsilon_2', 'p_d', 'epsilon_3',
                           'p_h', 'gamma_i_1', 'gamma_i_2', 'gamma_a_1', 'gamma_a_2',
                           'psi', 'alpha']
    cluster_specific_params = BaseMultiClusterVacConstructor.cluster_specific_params + ['eta']
    vaccine_specific_params = ['l', 's', 'h']

    def __init__(self, clusters, include_observed_states=True):
        if include_observed_states:
            self.observed_states = ['H_T', 'D_T']
            total_hospitalised = []
            total_recovered_from_hosptial = []
            total_detected_cases = []
        super().__init__(clusters)
        detected = 'p_d*epsilon_3'
        undetected_symptomatic = '(1-p_d)*epsilon_3'
        for vaccine_stage, vaccine_group in enumerate(self.vaccine_groups):
            l_v = 'l_' + vaccine_group

            s_v = 's_' + vaccine_group
            p_s_v = 'p_s * (1 -' + s_v +')'

            h_v = 'h_' + vaccine_group
            p_h_v = 'p_h * (1 -' + h_v +')'

            detected_at_hospital = p_h_v +'*' + detected
            detected_outside_hospital = '(1-' + p_h_v + ')*' + detected

            for cluster_i in self.clusters:

                eta_i = 'eta_' + cluster_i
                lambda_i_v = '(1-'+ l_v +')*lambda_' + cluster_i


                S_i_v = "S_" + cluster_i + '_' + vaccine_group
                E_i_v = "E_" + cluster_i + '_' + vaccine_group
                G_I_i_v = "G_I_" + cluster_i + '_' + vaccine_group
                G_A_i_v = "G_A_" + cluster_i + '_' + vaccine_group
                P_I_i_v = "P_I_" + cluster_i + '_' + vaccine_group
                P_A_i_v = "P_A_" + cluster_i + '_' + vaccine_group
                M_D_i_v = "M_D_" + cluster_i + '_' + vaccine_group
                M_I_i_v = "M_I_" + cluster_i + '_' + vaccine_group
                M_A_i_v = "M_A_" + cluster_i + '_' + vaccine_group
                F_D_i_v = "F_D_" + cluster_i + '_' + vaccine_group
                F_I_i_v = "F_I_" + cluster_i + '_' + vaccine_group
                F_A_i_v = "F_A_" + cluster_i + '_' + vaccine_group
                H_i_v = "H_" + cluster_i + '_' + vaccine_group
                R_i_v = "R_" + cluster_i + '_' + vaccine_group


                hospitalised = detected_at_hospital + '*' + P_I_i_v
                hospital_recovery = 'psi*' + H_i_v
                if include_observed_states:
                    total_hospitalised.append(hospitalised)
                    total_recovered_from_hosptial.append(hospital_recovery)
                    total_detected_cases.append(detected+'*'+P_I_i_v)
                self.transitions += [
                    Transition(origin=S_i_v, destination=E_i_v,
                               equation=lambda_i_v + '*' + S_i_v,
                               transition_type=TransitionType.T),
                    # E classes
                    Transition(origin=E_i_v, destination=G_I_i_v,
                               equation= p_s_v +'*epsilon_1*' + E_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=E_i_v, destination=G_A_i_v,
                               equation='(1-'+p_s_v+')*epsilon_1*' + E_i_v,
                               transition_type=TransitionType.T),
                    # G classes
                    Transition(origin=G_I_i_v, destination=P_I_i_v,
                               equation='epsilon_2*' + G_I_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=G_A_i_v, destination=P_A_i_v,
                               equation='epsilon_2*' + G_A_i_v,
                               transition_type=TransitionType.T),
                    # P classes
                    Transition(origin=P_I_i_v, destination=H_i_v,
                               equation=hospitalised,
                               transition_type=TransitionType.T),
                    Transition(origin=P_I_i_v, destination=M_D_i_v,
                               equation=detected_outside_hospital+ '*' + P_A_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=P_I_i_v, destination=M_I_i_v,
                               equation=undetected_symptomatic + '*' + P_A_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=P_A_i_v, destination=M_A_i_v,
                               equation='epsilon_3*' + P_A_i_v,
                               transition_type=TransitionType.T),
                    # M classes
                    Transition(origin=M_D_i_v, destination=F_D_i_v,
                               equation='gamma_i_1*' + M_D_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=M_I_i_v, destination=F_I_i_v,
                               equation='gamma_i_1*' + M_I_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=M_A_i_v, destination=F_A_i_v,
                               equation='gamma_a_1*' + M_A_i_v,
                               transition_type=TransitionType.T),
                    # F classes
                    Transition(origin=F_D_i_v, destination=R_i_v,
                               equation='gamma_i_2*' + F_D_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=F_I_i_v, destination=R_i_v,
                               equation='gamma_i_2*' + F_I_i_v,
                               transition_type=TransitionType.T),
                    Transition(origin=F_A_i_v, destination=R_i_v,
                               equation='gamma_a_2*' + F_A_i_v,
                               transition_type=TransitionType.T),

                    Transition(origin=H_i_v, destination=R_i_v,
                               equation=hospital_recovery,
                               transition_type=TransitionType.T),
                    Transition(origin=R_i_v, destination=S_i_v,
                               equation='alpha*'+ R_i_v,
                               transition_type=TransitionType.T)
                ]

        if include_observed_states:
            self.bd_list += [
                Transition(origin='D_T',
                           equation=' + '.join(total_detected_cases),
                           transition_type=TransitionType.B),
                Transition(origin='H_T',
                           equation=' + '.join(total_hospitalised),
                           transition_type=TransitionType.B),
                Transition(origin='H_T',
                           equation=' + '.join(total_recovered_from_hosptial),
                           transition_type=TransitionType.D)
            ]