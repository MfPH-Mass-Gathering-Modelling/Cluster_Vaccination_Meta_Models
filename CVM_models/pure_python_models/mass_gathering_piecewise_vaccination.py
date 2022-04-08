"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
from CVM_models.base_piecewise_vaccination import BaseSingleClusterVacModel



class MassGatheringModel(BaseSingleClusterVacModel):
    core_vaccine_groups = ['unvaccinated', 'first_dose', 'second_dose', 'third_dose']
    ve_delay_groups = ['first_dose', 'second_dose', 'third_dose']
    ve_wanning_groups = ['second_dose']
    states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2', 'H', 'D', 'R']
    dead_states = ['D']
    vaccinable_states = ['S', 'E', 'T', 'T_I', 'T_A', 'A_1', 'A_2', 'R']
    observed_states = ['H_total', 'D_total']
    infectious_states = ['T_I', 'T_A', 'A_1', 'A_2', 'I_1', 'I_2']
    symptomatic_states = ['I_1', 'I_2', 'H']

    def ode(self, y, t,
            beta,
            asymptomatic_tran_mod,
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
        foi = self.foi(y, beta, asymptomatic_tran_mod)
        y_deltas = np.zeros(self.num_states)
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
            now_infectious_and_LFD_sensitive_will_be_asymptomatic = (
                                                                                1 - vac_group_symptomatic_prop) * now_infectious_and_LFD_sensitive
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
            y_deltas[vac_group_states_index[
                'T_A']] += now_infectious_and_LFD_sensitive_will_be_asymptomatic - asymptomatic_prog_1
            y_deltas[vac_group_states_index[
                'T_I']] += now_infectious_and_LFD_sensitive_will_be_symptomatic - symptomatic_prog_1
            y_deltas[vac_group_states_index['A_1']] += asymptomatic_prog_1 - asymptomatic_prog_2
            y_deltas[vac_group_states_index['A_2']] += asymptomatic_prog_2 - asymptomatic_recovery
            y_deltas[vac_group_states_index['I_1']] += symptomatic_prog_1 - symptomatic_prog_2
            y_deltas[vac_group_states_index[
                'I_2']] += symptomatic_prog_2 - symptomatic_recovery - symptomatic_mortality - hospitalisation
            y_deltas[vac_group_states_index['H']] += hospitalisation - hospital_mortality - hospital_recovery
            overall_mortality = symptomatic_mortality + hospital_mortality
            y_deltas[vac_group_states_index['D']] += overall_mortality
            y_deltas[vac_group_states_index[
                'R']] += asymptomatic_recovery + symptomatic_recovery + hospital_recovery - waned_natural_immunity
            y_deltas[-2] += hospitalisation
            y_deltas[-1] += overall_mortality

        return y_deltas

    def jacobian(self, y, t,
            beta,
            asymptomatic_tran_mod,
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
        foi = self.foi(y, beta, asymptomatic_tran_mod)
        y_deltas = np.zeros(self.num_states)
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
            now_infectious_and_LFD_sensitive_will_be_asymptomatic = (
                                                                                1 - vac_group_symptomatic_prop) * now_infectious_and_LFD_sensitive
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
            y_deltas[vac_group_states_index[
                'T_A']] += now_infectious_and_LFD_sensitive_will_be_asymptomatic - asymptomatic_prog_1
            y_deltas[vac_group_states_index[
                'T_I']] += now_infectious_and_LFD_sensitive_will_be_symptomatic - symptomatic_prog_1
            y_deltas[vac_group_states_index['A_1']] += asymptomatic_prog_1 - asymptomatic_prog_2
            y_deltas[vac_group_states_index['A_2']] += asymptomatic_prog_2 - asymptomatic_recovery
            y_deltas[vac_group_states_index['I_1']] += symptomatic_prog_1 - symptomatic_prog_2
            y_deltas[vac_group_states_index[
                'I_2']] += symptomatic_prog_2 - symptomatic_recovery - symptomatic_mortality - hospitalisation
            y_deltas[vac_group_states_index['H']] += hospitalisation - hospital_mortality - hospital_recovery
            overall_mortality = symptomatic_mortality + hospital_mortality
            y_deltas[vac_group_states_index['D']] += overall_mortality
            y_deltas[vac_group_states_index[
                'R']] += asymptomatic_recovery + symptomatic_recovery + hospital_recovery - waned_natural_immunity
            y_deltas[-2] += hospitalisation
            y_deltas[-1] += overall_mortality

        return y_deltas