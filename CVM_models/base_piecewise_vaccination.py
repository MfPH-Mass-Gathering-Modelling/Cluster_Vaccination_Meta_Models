"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
import pandas as pd


class BaseSingleClusterVacModel:
    vaccination_groups = [
        'unvaccinated',
        'first_dose_non_effective',
        'first_dose_effective',
        'second_dose_non_effective',
        'second_dose_effective',
        'second_dose_waned',
        'third_dose_non_effective',
        'third_dose_effective'
    ]

    groups_waiting_for_vaccine_effectiveness = ['first_dose_non_effective', 'second_dose_non_effective',
                                                'third_dose_non_effective']
    groups_experiencing_waned_ve = ['second_dose_effective']
    groups_with_waned_immunity = ['second_dose_waned']
    states = []
    contactable_states = []
    vaccinable_states = []
    observed_states = []
    infectious_states = []
    symptomatic_states = []

    def __init__(self,
                 first_vac_dose, second_vac_dose, third_vac_dose,
                 ve_infection, ve_symptoms, ve_hospitalisation, ve_mortality):
        if isinstance(first_vac_dose, pd.Series):
            first_vac_dose = first_vac_dose.tolist()
        if isinstance(second_vac_dose, pd.Series):
            second_vac_dose = second_vac_dose.tolist()
        if isinstance(third_vac_dose, pd.Series):
            third_vac_dose = third_vac_dose.tolist()
        self.groups_loss_via_vaccination = {
            'unvaccinated': first_vac_dose,
            'first_dose_effective': second_vac_dose,
            'second_dose_waned': third_vac_dose
        }
        self._sorting_states()
        self._attaching_ve_dicts(ve_infection, ve_symptoms, ve_hospitalisation, ve_mortality)

    def _attaching_ve_dicts(self, ve_infection, ve_symptoms, ve_hospitalisation, ve_mortality):
        for name, var in vars().items():
            if name != "self":
                error_msg_end = name + ' should be a dictionary with keys being the same as the names of the vaccine groups and values being a float or int >=0 and <=1.'
                if not isinstance(var, dict):
                    raise TypeError(name + ' is not a dictionary. ' + error_msg_end)
                if set(var.keys()) != set(self.vaccination_groups):
                    raise ValueError(name + "'s keys are not in the list: " + ', '.join(self.vaccination_groups) +
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
        self.conactable_states_index = {}
        # populating index dictionaries
        index = 0
        for vaccine_group in self.vaccination_groups:
            self.conactable_states_index[vaccine_group] = {}
            self.state_index[vaccine_group] = {}
            self.infectious_symptomatic_index[vaccine_group] = {}
            self.infectious_asymptomatic_index[vaccine_group] = {}
            for state in self.states:
                self.state_index[vaccine_group][state] = index
                if state in self.infectious_and_symptomatic_states:
                    self.infectious_symptomatic_index[vaccine_group][state] = index
                if state in self.infectious_and_asymptomatic_states:
                    self.infectious_asymptomatic_index[vaccine_group][state] = index
                if state in self.contactable_states:
                    self.conactable_states_index[vaccine_group][state] = index
                index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.state_index['observed_states'][state] = index
            index += 1

        self.num_states = index

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
        conactable_states_index = self._nesteddictvalues(self.conactable_states_index)
        total_infectous_asymptomatic = y[infectious_asymptomatic_index].sum()
        total_infectous_symptomatic = y[infectious_symptomatic_index].sum()
        total_contactable_population = y[conactable_states_index].sum()
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

    def _nesteddictvalues(self, d):
        return [index for sub_d in d.values() for index in sub_d.values()]

    def vac_group_transfer(self, y, y_deltas, t,
                           inverse_effective_delay,
                           inverse_waning_immunity,
                           vacination_group
                           ):
        if vacination_group not in self.vaccination_groups:
            raise ValueError('vacination_group "' + vacination_group + '" not used in model or mispelled in code.' +
                             'Should be one of: "' + '", '.join(self.vaccination_groups) + '".')

        if vacination_group != self.vaccination_groups[-1]:
            # Unpack y elements relevant to this vaccination group.
            vac_group_states_index = self.state_index[vacination_group]
            index_of_next_vac_group = self.vaccination_groups.index(vacination_group) + 1
            next_vacination_group = self.vaccination_groups[index_of_next_vac_group]
            next_vac_group_states_index = self.state_index[next_vacination_group]

            # Lets deal with vaccinations first
            ## Then the groups being transfered to the next vaccination group
            if vacination_group in self.groups_loss_via_vaccination.keys():
                index_of_target_loss = int(t) + 1
                total_loss_via_vaccination = self.groups_loss_via_vaccination[vacination_group][index_of_target_loss]
                if total_loss_via_vaccination == 0:  # No point in calculations if no one is being vaccinated.
                    vacination_next_group_tranfer = {state: 0 for state in self.vaccinable_states}
                else:
                    total_vaccinable = 0
                    for state in self.vaccinable_states:
                        total_vaccinable += y[vac_group_states_index[state]]
                    vaccinable_populations = {state: y[vac_group_states_index[state]] for state in
                                              self.vaccinable_states}
                    vaccinable_propostions = {state: y[vac_group_states_index[state]] / total_vaccinable
                                              for state in self.vaccinable_states}
                    vacination_next_group_tranfer = {}
                    for state, population in vaccinable_populations.items():
                        proportion = vaccinable_propostions[state]
                        target_vaccination = total_loss_via_vaccination * proportion
                        if target_vaccination > population:
                            err_msg = ('Target vaccination population (' + str(target_vaccination) +
                                       ') > population ' + str(population) +
                                       ' of state "' + state +
                                       '" for vaccination group "' +
                                       vacination_group + '" at time ' + str(t) + '.')
                            raise ValueError(err_msg)

                        inst_loss_via_vaccine = self.instantaneous_transfer(target_vaccination, population, t)
                        vacination_next_group_tranfer[state] = np.nan_to_num(
                            population * (1 - np.exp(inst_loss_via_vaccine)))
            elif vacination_group in self.groups_waiting_for_vaccine_effectiveness:
                vacination_next_group_tranfer = {state: inverse_effective_delay * y[vac_group_states_index[state]]
                                                 for state in self.vaccinable_states}
            elif vacination_group in self.groups_experiencing_waned_ve:
                vacination_next_group_tranfer = {state: inverse_waning_immunity * y[vac_group_states_index[state]]
                                                 for state in self.vaccinable_states}
            else:
                raise ValueError(
                    'vacination_group "' + vacination_group + '" has no method of transfer to next vaccination group' +
                    'and is not the last vaccination group "' + self.vaccination_groups[-1] + '".')

            for state, transfering_pop in vacination_next_group_tranfer.items():
                y_deltas[vac_group_states_index[state]] -= transfering_pop
                y_deltas[next_vac_group_states_index[state]] += transfering_pop



