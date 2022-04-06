"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
import pandas as pd


class BaseSingleClusterVacModel:
    core_vaccine_groups = []
    ve_delay_groups = []
    ve_wanning_groups = []
    groups_waiting_for_vaccine_effectiveness = []
    states = []
    dead_states = []
    vaccinable_states = []
    observed_states = []
    infectious_states = []
    symptomatic_states = []

    def __init__(self, starting_population,
                 groups_loss_via_vaccination,
                 ve_dicts):
        self.starting_population = starting_population
        self.vaccine_groups = []
        for vaccine_group in self.core_vaccine_groups:
            if vaccine_group in self.ve_delay_groups:
                self.vaccine_groups.append(vaccine_group+'_delay')
                self.groups_waiting_for_vaccine_effectiveness.append(vaccine_group+'_delay')
            self.vaccine_groups.append(vaccine_group)
            if vaccine_group in self.ve_wanning_groups:
                self.vaccine_groups.append(vaccine_group + '_waned')
        groups_loss_via_vaccination_conv = {}
        for key, value in groups_loss_via_vaccination.items():
            if isinstance(value, pd.Series):
                new_value = value.tolist()
            else:
                new_value = value
            if key in self.ve_wanning_groups:
                new_key = key + '_waned'
            else:
                new_key = key
            groups_loss_via_vaccination_conv[new_key] = new_value
        self.groups_loss_via_vaccination = groups_loss_via_vaccination_conv
        self.derived_vaccination_rates = {key:{} for key in self.groups_loss_via_vaccination.keys()}
        self._sorting_states()
        self._attaching_ve_dicts(ve_dicts)


    def _attaching_ve_dicts(self, ve_dicts):
        for name, var in ve_dicts.items():
            error_msg_end = name + ' should be a dictionary with keys being the same as the names of the vaccine groups and values being a float or int >=0 and <=1.'
            if not isinstance(var, dict):
                raise TypeError(name + ' is not a dictionary. ' + error_msg_end)
            if set(var.keys()) != set(self.vaccine_groups):
                raise ValueError(name + "'s keys are not in the list: " + ', '.join(self.vaccine_groups) +
                                 ". " + error_msg_end)
            if not all([isinstance(item, (float, int)) for item in var.values()]):
                raise TypeError(name + ' values are not floats or ints. ' + error_msg_end)
            if not all([0 <= item <= 1 for item in var.values()]):
                raise ValueError(name + ' values are not >=0 and <=1. ' + error_msg_end)
            setattr(self, name, var)

    def _sorting_states(self):
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states]
        self.state_index = {}
        self.infectious_symptomatic_index = {}
        self.infectious_asymptomatic_index = {}
        self.vaccinable_states_index ={}
        self.dead_states_index = {}
        # populating index dictionaries
        index = 0
        for vaccine_group in self.vaccine_groups:
            self.dead_states_index[vaccine_group] = {}
            self.vaccinable_states_index[vaccine_group] = {}
            self.state_index[vaccine_group] = {}
            self.infectious_symptomatic_index[vaccine_group] = {}
            self.infectious_asymptomatic_index[vaccine_group] = {}
            for state in self.states:
                self.state_index[vaccine_group][state] = index
                if state in self.infectious_and_symptomatic_states:
                    self.infectious_symptomatic_index[vaccine_group][state] = index
                if state in self.infectious_and_asymptomatic_states:
                    self.infectious_asymptomatic_index[vaccine_group][state] = index
                if state in self.dead_states:
                    self.dead_states_index[vaccine_group][state] = index
                if state in self.vaccinable_states:
                    self.vaccinable_states_index[vaccine_group][state] = index
                index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.state_index['observed_states'][state] = index
            index += 1

        self.num_states = index

    def _nesteddictvalues(self, d):
        return [index for sub_d in d.values() for index in sub_d.values()]

    def foi(self, y, beta, asymptomatic_tran_mod):
        """Calculate force of infection (foi).

        Args:
            y (numpy.array): Value of current state variables.
            beta (float or int): Transmission coefficient.
            asymptomatic_tran_mod (float or int): Modification due to asymptomatic/pre-symptomatic state.

        Returns:
            float: Force of infection given state variables.
        """
        infectious_asymptomatic_index = self._nesteddictvalues(self.infectious_asymptomatic_index)
        infectious_symptomatic_index = self._nesteddictvalues(self.infectious_symptomatic_index)
        dead_states_index = self._nesteddictvalues(self.dead_states_index)
        total_infectous_asymptomatic = y[infectious_asymptomatic_index].sum()
        total_infectous_symptomatic = y[infectious_symptomatic_index].sum()
        total_contactable_population = self.starting_population - y[dead_states_index].sum()
        foi = (beta * (asymptomatic_tran_mod * total_infectous_asymptomatic +
                       total_infectous_symptomatic) / total_contactable_population)
        return foi

    def instantaneous_transfer(self, population_transitioning, population, t=None):
        if population_transitioning > population:
            error_msg = "population_transitioning (" + str(
                population_transitioning) + ") is greater than population (" + str(population)
            if t is None:
                error_msg += ').'
            else:
                error_msg += '), at time ' + str(t) + ').'

            raise ValueError(error_msg)
        numerator = population - population_transitioning
        denominator = population
        # growth should be 1, even if denominator and numerator are 0 and the natural log of 1 is 0 therefore.
        if numerator == 0 and denominator == 0:
            ans = 0
        else:
            growth = numerator / denominator
            growth = np.nan_to_num(growth, nan=1)
            ans = np.log(growth)
        return ans


    def vac_group_transfer(self, y, y_deltas, t,
                           inverse_effective_delay,
                           inverse_waning_immunity,
                           vaccine_group
                           ):
        if vaccine_group not in self.vaccine_groups:
            raise ValueError('vaccine_group "' + vaccine_group + '" not used in model or mispelled in code.' +
                             'Should be one of: "' + '", '.join(self.vaccine_groups) + '".')

        if vaccine_group != self.vaccine_groups[-1]:
            # Unpack y elements relevant to this vaccination group.
            vac_group_states_index = self.state_index[vaccine_group]
            index_of_next_vac_group = self.vaccine_groups.index(vaccine_group) + 1
            next_vaccine_group = self.vaccine_groups[index_of_next_vac_group]
            next_vac_group_states_index = self.state_index[next_vaccine_group]

            # Lets deal with vaccinations first
            ## Then the groups being transfered to the next vaccination group
            if vaccine_group in self.groups_loss_via_vaccination.keys():
                derived_vaccination_rates = self.derived_vaccination_rates[vaccine_group]
                if t not in derived_vaccination_rates.keys():
                    index_of_target_loss = int(t) + 1
                    total_loss_via_vaccination = self.groups_loss_via_vaccination[vaccine_group][index_of_target_loss]
                    if total_loss_via_vaccination == 0:  # No point in calculations if no one is being vaccinated.
                        derived_vaccination_rates[t] = 0
                    else:
                        vaccinable_states_index = list(self.vaccinable_states_index[vaccine_group].values())
                        total_vaccinable = y[vaccinable_states_index].sum()
                        inst_loss_via_vaccine = self.instantaneous_transfer(total_loss_via_vaccination, total_vaccinable, t)
                        derived_vaccination_rates[t] = 1 - np.exp(inst_loss_via_vaccine)
                vaccine_group_transfer = {state: derived_vaccination_rates[t] * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            elif vaccine_group in self.groups_waiting_for_vaccine_effectiveness:
                vaccine_group_transfer = {state: inverse_effective_delay * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            elif vaccine_group in self.ve_wanning_groups:
                vaccine_group_transfer = {state: inverse_waning_immunity * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            else:
                raise ValueError(
                    'vaccine_group "' + vaccine_group + '" has no method of transfer to next vaccination group ' +
                    'and is not the last vaccination group "' + self.vaccine_groups[-1] + '".')

            for state, transfering_pop in vaccine_group_transfer.items():
                y_deltas[vac_group_states_index[state]] -= transfering_pop
                y_deltas[next_vac_group_states_index[state]] += transfering_pop



