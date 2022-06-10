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
    states = ['S', 'E', 'G_I', 'G_A', 'P_I', 'P_A', 'M_D', 'M_I', 'M_A', 'F_D', 'F_I', 'F_A', 'H', 'R']
    observed_states = ['H_T', 'D_T']
    infectious_states = ['P_I', 'P_A', 'M_D', 'M_I', 'M_A', 'F_D', 'F_I', 'F_A']
    symptomatic_states = ['M_I','F_I','M_D','F_D','H']
    isolating_states = ['M_D','F_D']
    universal_param_names = ['beta', 'theta', 'kappa_D', 'epsilon_1', 'p_s',
                             'epsilon_2', 'epsilon_3', 'p_h']

    def ode(self, y, t, **arg_params):
        params = self.params_named_tuple(arg_params)
        foi = self.foi(y, params.beta, params.theta, params.kappa_D)
        y_deltas = np.zeros(self.num_state)
        for vaccine_group in self.vaccine_groups:
            self.vac_group_transfer(y, y_deltas, t, params.nu_d, params.nu_w, vaccine_group)
            vac_group_states_index = self.state_index[vaccine_group]
            # Infections
            infections = (1 - self.ve_infection[vaccine_group]) * foi * y[vac_group_states_index['S']]
            # progression to RT-pcr sensitivity
            prog_rtpcr = params.epsilon_1 * y[vac_group_states_index['E']]
            p_s_v = params.p_s * (1 - self.ve_symptoms[vaccine_group])
            prog_symptomatic_path = prog_rtpcr * p_s_v
            prog_asymptomatic_path = prog_rtpcr * (1-p_s_v)


            # progression to lfd/rapid antigen test sensitivity
            prog_LFD_sensitive_symptomatic_path = params.epsilon_2 * y[vac_group_states_index['G_I']]
            prog_LFD_sensitive_asymptomatic_path = params.epsilon_2 * y[vac_group_states_index['G_A']]


            # progression to mid-infection
            prog_mid_infection_symptomatic_path = params.epsilon_3 * y[vac_group_states_index['P_I']]
            prog_detected = params.p_d*prog_mid_infection_symptomatic_path
            prog_mid_infection_symptomatic_undetected = (1-params.p_d)*prog_mid_infection_symptomatic_path
            p_h_v = params.p_h*(1 - self.ve_hospitalisation[vaccine_group])
            prog_detected_at_hospital = p_h_v * prog_detected
            prog_detected_in_community = (1 - p_h_v) * prog_detected
            prog_mid_infection_asymptomatic_path = params.epsilon_3 * y[vac_group_states_index['P_A']]

            #Late infection
            prog_late_asymptomatic_stage = params.gamma_A_1 * y[vac_group_states_index['M_A']]
            prog_late_symptomatic_stage = params.gamma_I_1 * y[vac_group_states_index['M_I']]
            prog_late_detected_stage = params.gamma_I_1 * y[vac_group_states_index['M_D']]


            # recovery
            asymptomatic_recovery = params.gamma_A_2 * y[vac_group_states_index['F_A']]
            symptomatic_recovery = params.gamma_I_2 * y[vac_group_states_index['F_I']]
            detected_recovery = params.gamma_I_2 * y[vac_group_states_index['F_D']]
            hospital_recovery = params.gamma_H * y[vac_group_states_index['H']]

            # natural wanning imunity
            waned_natural_immunity = params.alpha * y[vac_group_states_index['R']]

            # Put together deltas
            y_deltas[vac_group_states_index['S']] += waned_natural_immunity - infections
            y_deltas[vac_group_states_index['E']] += infections - prog_rtpcr
            y[vac_group_states_index['G_I']] += prog_symptomatic_path - prog_LFD_sensitive_symptomatic_path
            y[vac_group_states_index['G_A']] += prog_asymptomatic_path - prog_LFD_sensitive_asymptomatic_path
            y[vac_group_states_index['P_I']] += prog_LFD_sensitive_symptomatic_path - prog_mid_infection_symptomatic_path
            y[vac_group_states_index['P_A']] += prog_LFD_sensitive_asymptomatic_path - prog_mid_infection_asymptomatic_path
            y[vac_group_states_index['M_A']] += prog_mid_infection_asymptomatic_path-prog_late_asymptomatic_stage
            y[vac_group_states_index['M_I']] += prog_mid_infection_symptomatic_undetected-prog_late_symptomatic_stage
            y[vac_group_states_index['M_D']] += prog_detected_in_community-prog_late_detected_stage
            y[vac_group_states_index['H']] += prog_detected_at_hospital - hospital_recovery
            y[vac_group_states_index['F_A']] += prog_late_asymptomatic_stage-asymptomatic_recovery
            y[vac_group_states_index['F_I']] += prog_late_asymptomatic_stage-symptomatic_recovery
            y[vac_group_states_index['F_D']] += prog_late_detected_stage-detected_recovery
            y_deltas[-2] += prog_detected_at_hospital - hospital_recovery
            y_deltas[-1] += prog_detected
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
