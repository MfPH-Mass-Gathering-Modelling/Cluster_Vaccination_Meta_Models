"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
from CVM_models.pure_python_models.base_piecewise_vaccination import BaseSingleClusterVacModel
import os
import json

# find this files directory so as to later find jacobians saved a json files
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath) +'/'


class MassGatheringModel(BaseSingleClusterVacModel):
    core_vaccine_groups = ['unvaccinated', 'first_dose', 'second_dose', 'third_dose']
    ve_delay_groups = ['first_dose', 'second_dose', 'third_dose']
    ve_wanning_groups = ['second_dose']
    states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2', 'H', 'D', 'R']
    dead_states = ['D']
    vaccinable_states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'R']
    observed_states = ['H_T', 'D_T']
    infectious_states = ['T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2']
    symptomatic_states = ['I_1', 'I_2', 'H']
    all_parameters = ['theta', 'epsilon_1', 'epsilon_2', 'epsilon_3', 'gamma_i_1', 'gamma_i_2',
                      'gamma_a_1', 'gamma_a_2', 'psi', 'rho', 'alpha', 'beta',
                      'l_unvaccinated', 'l_first_dose_delay', 'l_first_dose', 'l_second_dose_delay',
                      'l_second_dose', 'l_second_dose_waned', 'l_third_dose_delay', 'l_third_dose',
                      'r_unvaccinated', 'r_first_dose_delay', 'r_first_dose', 'r_second_dose_delay',
                      'r_second_dose', 'r_second_dose_waned', 'r_third_dose_delay', 'r_third_dose',
                      'h_unvaccinated', 'h_first_dose_delay', 'h_first_dose', 'h_second_dose_delay',
                      'h_second_dose', 'h_second_dose_waned', 'h_third_dose_delay', 'h_third_dose',
                      'm_unvaccinated', 'm_first_dose_delay', 'm_first_dose', 'm_second_dose_delay',
                      'm_second_dose', 'm_second_dose_waned', 'm_third_dose_delay', 'm_third_dose',
                      'N', 'eta', 'mu',
                      'nu_unvaccinated', 'nu_first_dose_delay', 'nu_first_dose', 'nu_second_dose_delay',
                      'nu_second_dose', 'nu_second_dose_waned', 'nu_third_dose_delay']

    def ode(self, y, t,
            beta,
            theta,
            inverse_effective_delay,
            inverse_waning_immunity,
            epsilon_1,
            epsilon_2,
            epsilon_3,
            rho,
            gamma_a_1,
            gamma_a_2,
            gamma_i_1,
            gamma_i_2,
            eta,
            mu,
            psi,
            alpha
            ):
        # params = (theta,
        #           inverse_effective_delay,
        #           inverse_waning_immunity,
        #           epsilon_1,
        #           epsilon_2,
        #           epsilon_3,
        #           rho,
        #           gamma_a_1,
        #           gamma_a_2,
        #           gamma_i_1,
        #           gamma_i_2,
        #           eta,
        #           mu,
        #           psi,
        #           alpha)
        # key = (params, tuple(y.tolist()), t)
        # if key in self.ode_calls_dict:
        #     y_deltas = self.ode_calls_dict[key]
        # else:
        foi = self.foi(y, beta, theta)
        y_deltas = np.zeros(self.num_state)
        for vaccine_group in self.vaccine_groups:
            self.vac_group_transfer(y, y_deltas, t, inverse_effective_delay, inverse_waning_immunity, vaccine_group)
            vac_group_states_index = self.state_index[vaccine_group]
            # Infections
            infections = (1 - self.ve_infection[vaccine_group]) * foi * y[vac_group_states_index['S']]
            # progression to RT-pcr sensitivity
            now_rtpcr_sensitive = epsilon_1 * y[vac_group_states_index['E']]
            # progression to lfd/rapid antigen test sensitivity
            now_infectious_and_LFD_sensitive = epsilon_2 * y[vac_group_states_index['T']]
            vac_group_symptomatic_prop = rho * (1 - self.ve_symptoms[vaccine_group])
            now_infectious_and_LFD_sensitive_will_be_symptomatic = vac_group_symptomatic_prop * now_infectious_and_LFD_sensitive
            now_infectious_and_LFD_sensitive_will_be_asymptomatic = (1 - vac_group_symptomatic_prop) * now_infectious_and_LFD_sensitive
            # progression with infection
            asymptomatic_prog_1 = epsilon_3 * y[vac_group_states_index['T_A']]
            symptomatic_prog_1 = epsilon_3 * y[vac_group_states_index['T_I']]
            asymptomatic_prog_2 = gamma_a_1 * y[vac_group_states_index['A_1']]
            symptomatic_prog_2 = gamma_i_1 * y[vac_group_states_index['I_1']]
            # recovery
            asymptomatic_recovery = gamma_a_2 * y[vac_group_states_index['A_2']]
            symptomatic_recovery = gamma_i_2 * y[vac_group_states_index['I_2']]
            hospital_recovery = psi * y[vac_group_states_index['H']]
            # hospitalisation
            hospitalisation = (1 - self.ve_hospitalisation[vaccine_group]) * eta * y[vac_group_states_index['I_2']]
            # mortality
            vac_group_symptomatic_mort = (1 - self.ve_mortality[vaccine_group]) * mu
            symptomatic_mortality = vac_group_symptomatic_mort * y[vac_group_states_index['I_2']]
            hospital_mortality = vac_group_symptomatic_mort * y[vac_group_states_index['H']]
            # natural wanning imunity
            waned_natural_immunity = alpha * y[vac_group_states_index['R']]

            # Put together deltas
            y_deltas[vac_group_states_index['S']] += waned_natural_immunity - infections
            y_deltas[vac_group_states_index['E']] += infections - now_rtpcr_sensitive
            y_deltas[vac_group_states_index['T']] += now_rtpcr_sensitive - now_infectious_and_LFD_sensitive
            y_deltas[vac_group_states_index['T_A']] += now_infectious_and_LFD_sensitive_will_be_asymptomatic - asymptomatic_prog_1
            y_deltas[vac_group_states_index['T_I']] += now_infectious_and_LFD_sensitive_will_be_symptomatic - symptomatic_prog_1
            y_deltas[vac_group_states_index['A_1']] += asymptomatic_prog_1 - asymptomatic_prog_2
            y_deltas[vac_group_states_index['A_2']] += asymptomatic_prog_2 - asymptomatic_recovery
            y_deltas[vac_group_states_index['I_1']] += symptomatic_prog_1 - symptomatic_prog_2
            y_deltas[vac_group_states_index['I_2']] += symptomatic_prog_2 - symptomatic_recovery - symptomatic_mortality - hospitalisation
            y_deltas[vac_group_states_index['H']] += hospitalisation - hospital_mortality - hospital_recovery
            overall_mortality = symptomatic_mortality + hospital_mortality
            y_deltas[vac_group_states_index['D']] += overall_mortality
            y_deltas[vac_group_states_index['R']] += asymptomatic_recovery + symptomatic_recovery + hospital_recovery - waned_natural_immunity
            y_deltas[-2] += hospitalisation - hospital_recovery - hospital_mortality
            y_deltas[-1] += overall_mortality
                # self.ode_calls_dict[key] = y_deltas

        return y_deltas

    def jacobian(self, y, t,
                 beta,
                 theta,
                 inverse_effective_delay,
                 inverse_waning_immunity,
                 epsilon_1,
                 epsilon_2,
                 epsilon_3,
                 rho,
                 gamma_a_1,
                 gamma_a_2,
                 gamma_i_1,
                 gamma_i_2,
                 eta,
                 mu,
                 psi,
                 alpha
                 ):
        if self.dok_jacobian is None:
            with open(dir_name + 'MG_model_jacobian.json') as json_file:
                self.dok_jacobian = json.load(json_file)
        # params = (theta,
        #           inverse_effective_delay,
        #           inverse_waning_immunity,
        #           epsilon_1,
        #           epsilon_2,
        #           epsilon_3,
        #           rho,
        #           gamma_a_1,
        #           gamma_a_2,
        #           gamma_i_1,
        #           gamma_i_2,
        #           eta,
        #           mu,
        #           psi,
        #           alpha)
        # key = (params, tuple(y.tolist()), t)
        # if key in self.ode_calls_dict:
        #     y_jacobian = self.jacobian_calls_dict[key]
        # else:
            state_value = y
            N = self.current_population(y)
            nu_unvaccinated = self.derived_vaccination_rates['unvaccinated'][t]
            nu_first_dose_delay = inverse_effective_delay
            nu_first_dose = self.derived_vaccination_rates['first_dose'][t]
            nu_second_dose_delay = inverse_effective_delay
            nu_second_dose = inverse_waning_immunity
            nu_waned = self.derived_vaccination_rates['second_dose_waned'][t]
            nu_third_dose_delay = inverse_effective_delay
            l_first_dose_delay = self.ve_infection['first_dose_delay']
            r_first_dose_delay = self.ve_symptoms['first_dose_delay']
            h_first_dose_delay = self.ve_hospitalisation['first_dose_delay']
            r_first_dose_delay = self.ve_mortality['first_dose_delay']

            l_first_dose = self.ve_infection['first_dose']
            r_first_dose = self.ve_symptoms['first_dose']
            h_first_dose = self.ve_hospitalisation['first_dose']
            r_first_dose = self.ve_mortality['first_dose']

            l_second_dose_delay = self.ve_infection['second_dose_delay']
            r_second_dose_delay = self.ve_symptoms['second_dose_delay']
            h_second_dose_delay = self.ve_hospitalisation['second_dose_delay']
            r_second_dose_delay = self.ve_mortality['second_dose_delay']

            l_second_dose = self.ve_infection['second_dose']
            r_second_dose = self.ve_symptoms['second_dose']
            h_second_dose = self.ve_hospitalisation['second_dose']
            r_second_dose = self.ve_mortality['second_dose']

            l_waned = self.ve_infection['second_dose_waned']
            r_waned = self.ve_symptoms['second_dose_waned']
            h_waned = self.ve_hospitalisation['second_dose_waned']
            r_waned = self.ve_mortality['second_dose_waned']

            l_third_dose_delay = self.ve_infection['third_dose_delay']
            r_third_dose_delay = self.ve_symptoms['third_dose_delay']
            h_third_dose_delay = self.ve_hospitalisation['third_dose_delay']
            r_third_dose_delay = self.ve_mortality['third_dose_delay']

            l_third_dose = self.ve_infection['third_dose']
            r_third_dose = self.ve_symptoms['third_dose']
            h_third_dose = self.ve_hospitalisation['third_dose']
            r_third_dose = self.ve_mortality['third_dose']
            y_jacobian = np.zeros(self.num_state,self.num_state)
            # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

            for coord, value in self.dok_jacobian.items():
                y_jacobian[eval(coord)] = eval(value)
            # self.jacobian_calls_dict[key] = y_jacobian
        return y_jacobian
    
    def diff_jacobian(self, y, t,
                 beta,
                 theta,
                 inverse_effective_delay,
                 inverse_waning_immunity,
                 epsilon_1,
                 epsilon_2,
                 epsilon_3,
                 rho,
                 gamma_a_1,
                 gamma_a_2,
                 gamma_i_1,
                 gamma_i_2,
                 eta,
                 mu,
                 psi,
                 alpha
                 ):
        if self.dok_diff_jacobian is None:
            with open(dir_name + 'MG_model_diff_jacobian.json') as json_file:
                self.dok_diff_jacobian = json.load(json_file)
        state_value = y
        N = self.current_population(y)
        nu_unvaccinated = self.derived_vaccination_rates['unvaccinated'][t]
        nu_first_dose_delay = inverse_effective_delay
        nu_first_dose = self.derived_vaccination_rates['first_dose'][t]
        nu_second_dose_delay = inverse_effective_delay
        nu_second_dose = inverse_waning_immunity
        nu_waned = self.derived_vaccination_rates['second_dose_waned'][t]
        nu_third_dose_delay = inverse_effective_delay
        l_first_dose_delay = self.ve_infection['first_dose_delay']
        r_first_dose_delay = self.ve_symptoms['first_dose_delay']
        h_first_dose_delay = self.ve_hospitalisation['first_dose_delay']
        r_first_dose_delay = self.ve_mortality['first_dose_delay']

        l_first_dose = self.ve_infection['first_dose']
        r_first_dose = self.ve_symptoms['first_dose']
        h_first_dose = self.ve_hospitalisation['first_dose']
        r_first_dose = self.ve_mortality['first_dose']

        l_second_dose_delay = self.ve_infection['second_dose_delay']
        r_second_dose_delay = self.ve_symptoms['second_dose_delay']
        h_second_dose_delay = self.ve_hospitalisation['second_dose_delay']
        r_second_dose_delay = self.ve_mortality['second_dose_delay']

        l_second_dose = self.ve_infection['second_dose']
        r_second_dose = self.ve_symptoms['second_dose']
        h_second_dose = self.ve_hospitalisation['second_dose']
        r_second_dose = self.ve_mortality['second_dose']

        l_waned = self.ve_infection['second_dose_waned']
        r_waned = self.ve_symptoms['second_dose_waned']
        h_waned = self.ve_hospitalisation['second_dose_waned']
        r_waned = self.ve_mortality['second_dose_waned']

        l_third_dose_delay = self.ve_infection['third_dose_delay']
        r_third_dose_delay = self.ve_symptoms['third_dose_delay']
        h_third_dose_delay = self.ve_hospitalisation['third_dose_delay']
        r_third_dose_delay = self.ve_mortality['third_dose_delay']

        l_third_dose = self.ve_infection['third_dose']
        r_third_dose = self.ve_symptoms['third_dose']
        h_third_dose = self.ve_hospitalisation['third_dose']
        r_third_dose = self.ve_mortality['third_dose']
        y_diff_jacobian = np.zeros(self.num_state**2,self.num_state)
        # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

        for coord, value in self.dok_diff_jacobian.items():
            y_diff_jacobian[eval(coord)] = eval(value)

        return y_diff_jacobian

    def gradient(self, y, t,
                 beta,
                 theta,
                 inverse_effective_delay,
                 inverse_waning_immunity,
                 epsilon_1,
                 epsilon_2,
                 epsilon_3,
                 rho,
                 gamma_a_1,
                 gamma_a_2,
                 gamma_i_1,
                 gamma_i_2,
                 eta,
                 mu,
                 psi,
                 alpha
                 ):
        if self.dok_gradient is None:
            with open(dir_name + 'MG_model_ode_gradient.json') as json_file:
                self.dok_gradient = json.load(json_file)
        state_value = y
        N = self.current_population(y)
        nu_unvaccinated = self.derived_vaccination_rates['unvaccinated'][t]
        nu_first_dose_delay = inverse_effective_delay
        nu_first_dose = self.derived_vaccination_rates['first_dose'][t]
        nu_second_dose_delay = inverse_effective_delay
        nu_second_dose = inverse_waning_immunity
        nu_waned = self.derived_vaccination_rates['second_dose_waned'][t]
        nu_third_dose_delay = inverse_effective_delay
        l_first_dose_delay = self.ve_infection['first_dose_delay']
        r_first_dose_delay = self.ve_symptoms['first_dose_delay']
        h_first_dose_delay = self.ve_hospitalisation['first_dose_delay']
        r_first_dose_delay = self.ve_mortality['first_dose_delay']

        l_first_dose = self.ve_infection['first_dose']
        r_first_dose = self.ve_symptoms['first_dose']
        h_first_dose = self.ve_hospitalisation['first_dose']
        r_first_dose = self.ve_mortality['first_dose']

        l_second_dose_delay = self.ve_infection['second_dose_delay']
        r_second_dose_delay = self.ve_symptoms['second_dose_delay']
        h_second_dose_delay = self.ve_hospitalisation['second_dose_delay']
        r_second_dose_delay = self.ve_mortality['second_dose_delay']

        l_second_dose = self.ve_infection['second_dose']
        r_second_dose = self.ve_symptoms['second_dose']
        h_second_dose = self.ve_hospitalisation['second_dose']
        r_second_dose = self.ve_mortality['second_dose']

        l_waned = self.ve_infection['second_dose_waned']
        r_waned = self.ve_symptoms['second_dose_waned']
        h_waned = self.ve_hospitalisation['second_dose_waned']
        r_waned = self.ve_mortality['second_dose_waned']

        l_third_dose_delay = self.ve_infection['third_dose_delay']
        r_third_dose_delay = self.ve_symptoms['third_dose_delay']
        h_third_dose_delay = self.ve_hospitalisation['third_dose_delay']
        r_third_dose_delay = self.ve_mortality['third_dose_delay']

        l_third_dose = self.ve_infection['third_dose']
        r_third_dose = self.ve_symptoms['third_dose']
        h_third_dose = self.ve_hospitalisation['third_dose']
        r_third_dose = self.ve_mortality['third_dose']
        y_gradient = np.zeros(self.num_state, self.num_param)
        # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

        for coord, value in self.dok_gradient.items():
            y_gradient[eval(coord)] = eval(value)
        return y_gradient

    def grad_jacobian(self, y, t,
                      beta,
                      theta,
                      inverse_effective_delay,
                      inverse_waning_immunity,
                 epsilon_1,
                 epsilon_2,
                 epsilon_3,
                 rho,
                 gamma_a_1,
                 gamma_a_2,
                 gamma_i_1,
                 gamma_i_2,
                 eta,
                 mu,
                 psi,
                 alpha
                 ):
        if self.dok_gradients_jacobian is None:
            with open(dir_name + 'MG_model_gradients_jacobian.json') as json_file:
                dok_gradients_jacobian = json.load(json_file)
        state_value = y
        N = self.current_population(y)
        nu_unvaccinated = self.derived_vaccination_rates['unvaccinated'][t]
        nu_first_dose_delay = inverse_effective_delay
        nu_first_dose = self.derived_vaccination_rates['first_dose'][t]
        nu_second_dose_delay = inverse_effective_delay
        nu_second_dose = inverse_waning_immunity
        nu_waned = self.derived_vaccination_rates['second_dose_waned'][t]
        nu_third_dose_delay = inverse_effective_delay
        l_first_dose_delay = self.ve_infection['first_dose_delay']
        r_first_dose_delay = self.ve_symptoms['first_dose_delay']
        h_first_dose_delay = self.ve_hospitalisation['first_dose_delay']
        r_first_dose_delay = self.ve_mortality['first_dose_delay']

        l_first_dose = self.ve_infection['first_dose']
        r_first_dose = self.ve_symptoms['first_dose']
        h_first_dose = self.ve_hospitalisation['first_dose']
        r_first_dose = self.ve_mortality['first_dose']

        l_second_dose_delay = self.ve_infection['second_dose_delay']
        r_second_dose_delay = self.ve_symptoms['second_dose_delay']
        h_second_dose_delay = self.ve_hospitalisation['second_dose_delay']
        r_second_dose_delay = self.ve_mortality['second_dose_delay']

        l_second_dose = self.ve_infection['second_dose']
        r_second_dose = self.ve_symptoms['second_dose']
        h_second_dose = self.ve_hospitalisation['second_dose']
        r_second_dose = self.ve_mortality['second_dose']

        l_waned = self.ve_infection['second_dose_waned']
        r_waned = self.ve_symptoms['second_dose_waned']
        h_waned = self.ve_hospitalisation['second_dose_waned']
        r_waned = self.ve_mortality['second_dose_waned']

        l_third_dose_delay = self.ve_infection['third_dose_delay']
        r_third_dose_delay = self.ve_symptoms['third_dose_delay']
        h_third_dose_delay = self.ve_hospitalisation['third_dose_delay']
        r_third_dose_delay = self.ve_mortality['third_dose_delay']

        l_third_dose = self.ve_infection['third_dose']
        r_third_dose = self.ve_symptoms['third_dose']
        h_third_dose = self.ve_hospitalisation['third_dose']
        r_third_dose = self.ve_mortality['third_dose']
        y_gradient_jacobian = np.zeros(self.num_state, self.num_param)
        # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

        for coord, value in self.dok_gradients_jacobian.items():
            y_gradient_jacobian[eval(coord)] = eval(value)
        return y_gradient_jacobian
