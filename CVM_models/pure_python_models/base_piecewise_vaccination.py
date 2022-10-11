"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
import pandas as pd
from numbers import Number
import json
import scipy
import functools


def _nesteddictvalues(d):
    return [index for sub_d in d.values() for index in sub_d.values()]
class BaseScipyClusterVacModel:
    states = []
    dead_states = []
    observed_states = []
    infected_states = []
    hospitalised_states = []
    infectious_states = []
    symptomatic_states = []
    isolating_states = []
    transmission_term = 'beta'
    population_term = 'N'
    transmission_cluster_specfic = False
    isolation_modifier = None
    isolation_cluster_specfic = False
    asymptomatic_transmission_modifier = None
    non_transmission_universal_params = []
    non_transmission_cluster_specific_params = [] # does not include transmission term beta.
    vaccine_specific_params = []



    def __init__(self, group_structure):
        self.all_parameters = set(self.non_transmission_universal_params)
        if self.asymptomatic_transmission_modifier is not None:
            self.all_parameters.add(self.asymptomatic_transmission_modifier)
        self.gen_group_structure(group_structure)
        self.all_parameters.update([param + '_' + vaccine_group
                                    for vaccine_group in self.vaccine_groups
                                    for param in self.vaccine_specific_params
                                    if param not in self.non_transmission_cluster_specific_params])
        self.all_parameters.update([param + '_' + cluster
                                    for cluster in self.clusters
                                    for param in self.non_transmission_cluster_specific_params
                                    if param not in self.vaccine_specific_params])
        self.all_parameters.update([param + '_' + cluster + '_' + vaccine_group
                                    for cluster in self.clusters
                                    for vaccine_group in self.vaccine_groups
                                    for param in self.vaccine_specific_params
                                    if param in self.non_transmission_cluster_specific_params])
        if self.transmission_cluster_specfic:
            self.transmission_to_terms = {cluster_i: {self.transmission_term: [],
                                                      self.population_term: []
                                                      } for cluster_i in self.clusters}

            self.transmission_from_terms = {cluster_j: {self.transmission_term: [],
                                                        self.population_term: []
                                                        } for cluster_j in self.clusters}
            for cluster_i in self.clusters:
                term = self.population_term + '_' + cluster_i
                self.all_parameters.add(term)
                self.transmission_to_terms[cluster_i][self.population_term].append(term)
                for cluster_j in self.clusters:
                    term = self.transmission_term + '_' + cluster_i + '_' + cluster_j
                    self.all_parameters.add(term)
                    self.transmission_to_terms[cluster_i][self.transmission_term].append(term)
                    self.transmission_from_terms[cluster_j][self.transmission_term].append(term)
        if self.isolation_cluster_specfic:
            if self.isolation_modifier is None:
                raise AssertionError('isolation_modifier must be specifed to be considered cluster specific')
            if not self.transmission_cluster_specfic:
                raise AssertionError('isolation being cluster specific is only supported when transmission is cluster specific.')
            self.all_parameters.update([self.isolation_modifier + '_' + cluster
                                        for cluster in self.clusters])
        self.all_parameters = sorted(self.all_parameters)
        non_piece_wise_params_names = set(self.all_parameters)-set(self.params_estimated_via_piecewise_method)
        self.non_piece_wise_params_names = sorted(list(non_piece_wise_params_names))
        self._sorting_states()
        # self.ode_calls_dict = {}
        # self.jacobian_calls_dict = {}
        self.dok_jacobian = None
        self.dok_diff_jacobian = None
        self.dok_gradient = None
        self.dok_gradients_jacobian = None
        self._lossObj = None
        self._parameters = None
        self._initial_values = None
        self._initial_time = None
        self._observations = None
        self._observation_state_index = None
        self._targetParam = None
        self._time_frame = None
        self.num_param = len(self.all_parameters)
        self.num_piece_wise_params = len(self.params_estimated_via_piecewise_method)
        self.num_non_piece_wise_params = self.num_param - self.num_piece_wise_params


    def get_transmission_terms_between(self, clusters):
        population_terms = []
        transmission_terms = []
        for cluster_i in clusters:
            for cluster_j in clusters:
                transmission_term = self.transmission_term + '_' + cluster_i + '_' + cluster_j
                population_term = self.population_term + '_' + cluster_i + '_' + cluster_j
                if transmission_term not in transmission_terms:
                    transmission_terms.append(transmission_term)
                    population_terms.append(population_term)
        return {self.transmission_term: transmission_terms, self.population_term: population_terms}


    def gen_group_structure(self, group_structure):
        self.params_estimated_via_piecewise_method = []
        self.group_transfer_dict = {}
        self.group_transition_params_dict = {}
        if isinstance(group_structure, dict):
            self.vaccine_groups = group_structure['vaccine groups']
            self.clusters = group_structure['clusters']
        elif isinstance(group_structure, list):
            self.vaccine_groups = []
            self.clusters = []
            for group_transfer in group_structure:
                cluster = group_transfer['from_cluster']
                if cluster not in self.clusters:
                    self.clusters.append(cluster)
                if cluster not in self.group_transfer_dict:
                    self.group_transfer_dict[cluster] = {}
                vaccine_group = group_transfer['from_vaccine_group']
                if vaccine_group not in self.vaccine_groups:
                    self.vaccine_groups.append(vaccine_group)
                if vaccine_group not in self.group_transfer_dict[cluster]:
                    self.group_transfer_dict[cluster][vaccine_group] = []
                to_cluster = group_transfer['to_cluster']
                if to_cluster not in self.clusters:
                    self.clusters.append(to_cluster)
                to_vaccine_group = group_transfer['to_vaccine_group']
                if to_vaccine_group not in self.vaccine_groups:
                    self.vaccine_groups.append(to_vaccine_group)

                if group_transfer['states']=='all':
                    group_transfer['states'] = self.states
                else:
                    for state in group_transfer['states']:
                        self._check_string_in_list_strings(state, 'states')
                parameter = group_transfer['parameter']
                if not isinstance(parameter, str):
                    raise TypeError(str(parameter) + ' should be of type string.')
                if parameter not in self.group_transition_params_dict:
                    self.group_transition_params_dict[parameter] = []
                entry = {key: value for key, value in
                         group_transfer.items()
                         if key != 'parameter'}
                self.group_transition_params_dict[parameter].append(entry)
                self.all_parameters.add(parameter)
                if 'piecewise targets' in group_transfer:
                    self.params_estimated_via_piecewise_method.append(parameter)
                    if isinstance(group_transfer['piecewise targets'], pd.Series):
                        group_transfer['piecewise targets'] = group_transfer['piecewise targets'].tolist()
                entry = {key: value
                         for key, value in group_transfer.items()
                         if key not in ['from_cluster', 'from_vaccine_group']}
                self.group_transfer_dict[cluster][vaccine_group].append(entry)


    def group_transfer(self, y, y_deltas, t,
                       from_cluster,
                       from_vaccine_group,
                       params
                       ):
        if from_cluster in self.group_transfer_dict:
            if from_vaccine_group in self.group_transfer_dict[from_cluster]:
                group_transfers = self.group_transfer_dict[from_cluster][from_vaccine_group]
                from_index_dict = self.state_index[from_cluster][from_vaccine_group]
                for group_transfer in group_transfers:
                    parameter = group_transfer['parameter']
                    if 'piecewise targets' in group_transfer:
                        if t in self.piecewise_est_param_values[parameter]:
                            param_val = self.piecewise_est_param_values[parameter][t]
                        else:
                            index_of_t = int(t) + 1
                            total_being_tranfered = group_transfer['piecewise targets'][index_of_t]
                            if total_being_tranfered == 0:  # No point in calculations if no one is being vaccinated.
                                param_val = 0
                            else:
                                from_states_index = [from_index_dict[state] for state in group_transfer['states']]
                                total_avialable = y[from_states_index].sum()
                                inst_loss_via_vaccine = self.instantaneous_transfer(total_being_tranfered,
                                                                                    total_avialable, t)
                                param_val = 1 - np.exp(inst_loss_via_vaccine)

                            self.piecewise_est_param_values[parameter][t] = param_val
                    else:
                        param_val = params[parameter]

                    to_cluster = group_transfer['to_cluster']
                    to_vaccine_group = group_transfer['to_vaccine_group']
                    to_index_dict = self.state_index[to_cluster][to_vaccine_group]
                    for state in group_transfer['states']:
                        from_index = from_index_dict[state]
                        to_index = to_index_dict[state]
                        transferring = param_val * y[from_index]
                        y_deltas[from_index] -= transferring
                        y_deltas[to_index] += transferring


    def _check_string_in_list_strings(self, string, list_strings):
        if not isinstance(string,str):
            raise TypeError(str(string) +' should be of type string.')

        check_list = eval('self.' + list_strings)
        if string not in check_list:
            raise ValueError(string + ' is not one of the predefined model ' + list_strings + ': ' +
                             ','.join(check_list[:-1]) + ' and ' + check_list[:-1] +'.')


    def _sorting_states(self):
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states and
                                                  state not in self.isolating_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states and
                                                   state not in self.isolating_states]
        self.isolating_and_symptomatic_states = [state for state in self.infectious_states
                                                 if state in self.symptomatic_states and
                                                 state in self.isolating_states]
        self.isolating_and_asymptomatic_states = [state for state in self.infectious_states
                                                  if state not in self.symptomatic_states and
                                                  state in self.isolating_states]
        self.all_states_index = {}
        self.state_index = {}
        self.infectious_symptomatic_indexes = {}
        self.infectious_asymptomatic_indexes = {}
        self.isolating_asymptomatic_indexes = {}
        self.isolating_symptomatic_indexes = {}
        self.infected_states_index_list = []
        self.hospitalised_states_index_list = []
        self.dead_states_index_list = []
        # populating index dictionaries
        index = 0
        for cluster in self.clusters:
            self.state_index[cluster] = {}
            self.infectious_symptomatic_indexes[cluster] = []
            self.infectious_asymptomatic_indexes[cluster] = []
            self.isolating_asymptomatic_indexes[cluster] = []
            self.isolating_symptomatic_indexes[cluster] = []
            for vaccine_group in self.vaccine_groups:
                self.state_index[cluster][vaccine_group] = {}
                for state in self.states:
                    self.all_states_index[state+'_'+cluster+'_'+vaccine_group] = index
                    self.state_index[cluster][vaccine_group][state] = index
                    if state in self.infectious_and_symptomatic_states:
                        self.infectious_symptomatic_indexes[cluster].append(index)
                    if state in self.infectious_and_asymptomatic_states:
                        self.infectious_asymptomatic_indexes[cluster].append(index)
                    if state in self.isolating_and_symptomatic_states:
                        self.isolating_symptomatic_indexes[cluster].append(index)
                    if state in self.isolating_and_asymptomatic_states:
                        self.isolating_asymptomatic_indexes[cluster].append(index)
                    if state in self.infected_states:
                        self.infected_states_index_list.append(index)
                    if state in self.hospitalised_states:
                        self.hospitalised_states_index_list.append(index)
                    if state in self.dead_states:
                        self.dead_states_index_list.append(index)
                    index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.all_states_index[state] = index
            self.state_index['observed_states'][state] = index
            index += 1

        self.num_state = index
        for transfer_info in self.group_transition_params_dict.values():
            for transfer_info_entry in transfer_info:
                cluster = transfer_info_entry['from_cluster']
                vaccine_group = transfer_info_entry['from_vaccine_group']
                states_dict = self.state_index[cluster][vaccine_group]
                to_cluster = transfer_info_entry['to_cluster']
                to_vaccine_group = transfer_info_entry['to_vaccine_group']
                to_states_dict = self.state_index[to_cluster][to_vaccine_group]
                state_selection = transfer_info_entry['states']
                if state_selection == 'all':
                    transfer_info_entry['from_index'] = [states_dict.values()]
                    transfer_info_entry['to_index'] = [to_states_dict.values()]
                else:
                    transfer_info_entry['from_index'] = [states_dict[state] for state in state_selection]
                    transfer_info_entry['to_index'] = [to_states_dict[state] for state in state_selection]



    
    def get_clusters_indexes(self, clusters):
        indexes = []
        for cluster in clusters:
            indexes += _nesteddictvalues(self.state_index[cluster])
        return indexes

    def get_vaccine_group_indexes(self, vaccine_groups):
        indexes = []
        for cluster in self.clusters:
            for vaccine_group, sub_dict in self.state_index[cluster].items():
                if vaccine_group in vaccine_groups:
                    indexes += [sub_dict.values()]
        return indexes
        

    def foi(self, y, parameters):
        """Calculate force of infection (foi).

        Args:
            y (numpy.array): Value of current state variables.
            beta (float or int): Transmission coefficient.
            asymptomatic_tran_mod (float or int): Modification due to asymptomatic/pre-symptomatic state.
            isolation_mod: Modification due to isolation.

        Returns:
            float: Force of infection given state variables.
        """
        if self.asymptomatic_transmission_modifier is not None:
            asymptomatic_transmission_modifier = parameters[self.asymptomatic_transmission_modifier]
        else:
            asymptomatic_transmission_modifier = 1

        if self.transmission_cluster_specfic:
            fois = {}
            for cluster_i in self.clusters:
                foi = 0
                contactable_population = parameters[self.population_term + '_' + cluster_i]
                for cluster_j in self.clusters:
                    beta = parameters[self.transmission_term + '_' + cluster_i + '_' + cluster_j]

                    if self.isolation_modifier is not None:
                        if self.isolation_cluster_specfic:
                            isolation_mod = parameters[self.isolation_modifier + '_' +cluster_j]
                        else:
                            isolation_mod = parameters[self.isolation_modifier]
                    else:
                        isolation_mod = 1
                    if beta > 0:
                        total_asymptomatic = (asymptomatic_transmission_modifier *
                                              y[self.infectious_asymptomatic_indexes[cluster_j]].sum())
                        total_symptomatic = y[self.infectious_symptomatic_indexes[cluster_j]].sum()
                        total_isolating_asymptomatic = (isolation_mod * asymptomatic_transmission_modifier *
                                                        y[self.isolating_asymptomatic_indexes[cluster_j]].sum())
                        total_isolating_symptomatic = (isolation_mod *
                                                       y[self.isolating_symptomatic_indexes[cluster_j]].sum())
                        full_contribution = sum([total_asymptomatic, total_symptomatic,
                                                 total_isolating_asymptomatic, total_isolating_symptomatic])

                        foi += beta * full_contribution / contactable_population

                fois[cluster_i] = foi
        else:
            if self.isolation_modifier is not None:
                isolation_mod = parameters[self.isolation_modifier]
            else:
                isolation_mod = 1
            infectious_symptomatic_indexes = _nesteddictvalues(self.infectious_symptomatic_indexes)
            infectious_and_asymptomatic_indexes = _nesteddictvalues(self.infectious_asymptomatic_indexes)
            isolating_asymptomatic_indexes = _nesteddictvalues(self.isolating_asymptomatic_indexes)
            isolating_symptomatic_indexes = _nesteddictvalues(self.isolating_symptomatic_indexes)
            total_asymptomatic = asymptomatic_transmission_modifier *y[infectious_and_asymptomatic_indexes].sum()
            total_symptomatic = y[infectious_symptomatic_indexes].sum()
            total_isolating_asymptomatic = isolation_mod*asymptomatic_transmission_modifier *y[isolating_asymptomatic_indexes].sum()
            total_isolating_symptomatic = isolation_mod*y[isolating_symptomatic_indexes].sum()
            full_contribution = sum([total_asymptomatic,total_symptomatic,
                                     total_isolating_asymptomatic,total_isolating_symptomatic])
            total_contactable_population = self.current_population(y)
            fois = parameters[self.transmission_term] * full_contribution / parameters[self.population_term]
        return fois

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

    @property
    def parameters(self):
        """
        Returns
        -------
        list
            A list which contains tuple of two elements,
            (:mod:`sympy.core.symbol`, numeric)

        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if not isinstance(parameters, dict):
            raise TypeError('Currently non non_piecewise_params must be entered as a dict.')
        # we assume that the key of the dictionary is a string and
        # the value can be a single value or a distribution

        for param_name, value in parameters.items():
            if param_name not in self.all_parameters:
                raise ValueError(param_name + ' is not a name given to a parameter for this model.')
            if param_name in self.params_estimated_via_piecewise_method:
                raise AssertionError(param_name + ' was set as a parameter to be estimated via piecewise estimiation ' +
                                     'at the initialization of this model.')
            if not isinstance(value, Number):
                raise TypeError(param_name + ' is not a number type.')
        params_not_given = [param for param in self.all_parameters
                            if param not in
                            list(parameters.keys()) + self.params_estimated_via_piecewise_method]
        if params_not_given:
            raise Exception(', '.join(params_not_given) +
                            " are/is missing from parameters for model (see self.all_parameters).")
        # this must be sorted alphanumerically.
        self._parameters = {key: value for key, value in sorted(parameters.items())}

    def check_all_params_represented(self):
        check_list = (list(self.parameters.keys()) +
                      self.params_estimated_via_piecewise_method)
        for param in self.all_parameters:
            if param not in check_list:
                raise AssertionError(param +
                                     'has not been assigned a value or set up for piecewise estimation.')


    def integrate(self, x0, t, full_output=False, called_in_fitting=False, **kwargs_to_pass_to_odeint):
        '''
        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>` using
        :class:`DeterministicOde <pygom.model.DeterministicOde>`.

        Parameters
        ----------
        t: array like
            the time points including initial time
        full_output: bool, optional
            If the additional information from the integration is required

        '''
        if not called_in_fitting: # If fitting model these checks should be done in fitting method.
            # This would avoid unnecessary error checks.
            self.check_all_params_represented()
        self.piecewise_est_param_values = {param: {} for param in self.params_estimated_via_piecewise_method}
        # INTEGRATE!!! (shout it out loud, in Dalek voice)
        # determine the number of output we want
        args = tuple(self.parameters.values())
        if self.dok_jacobian is None: # May or may not of defined the models Jacobian
            solution, output = scipy.integrate.odeint(self.ode,
                                                      x0, t, args=args,
                                                      full_output=True,
                                                      **kwargs_to_pass_to_odeint)
        else:
            solution, output = scipy.integrate.odeint(self.ode,
                                                      x0, t, args=args,
                                                      Dfun=self.jacobian,
                                                      full_output=True,
                                                      **kwargs_to_pass_to_odeint)
        if full_output == True:
            # have both
            return solution, output
        else:
            return solution

    #%% JACOBIAN MATRICIES ETC OF MODELS

    def load_dok_jacobian(self, json_file):
        with open(json_file) as json_file:
            self.dok_jacobian = json.load(json_file)


    def _sorting_params(self, parameters):
        return dict(zip(self.non_piece_wise_params_names, parameters))

    def jacobian(self, y, t, *parameters):
        if self.dok_jacobian is None:
            raise AssertionError('Dictionary of Keys (DOK) version of jacobian needs to be loaded first.' +
                                 'Call method load_dok_jacobian.')
        parameters = self._sorting_params(parameters)
        N = self.current_population(y)
        y_jacobian = np.zeros(self.num_state, self.num_state)
        for coord, value in self.dok_jacobian.items():
            y_jacobian[eval(coord)] = eval(value)

        return y_jacobian

    def load_dok_diff_jacobian(self, json_file):
        with open(json_file) as json_file:
            self.dok_diff_jacobian = json.load(json_file)


    def diff_jacobian(self, y, t, parameters):
        if self.dok_diff_jacobian is None:
            raise AssertionError('Dictionary of Keys (DOK) version of matrix needs to be loaded first.' +
                                 'Call method load_dok_diff_jacobian.')
        parameters = self._sorting_params(parameters)
        N = self.current_population(y)
        y_diff_jacobian = np.zeros(self.num_state ** 2, self.num_state)
        for coord, value in self.dok_diff_jacobian.items():
            y_diff_jacobian[eval(coord)] = eval(value)

        return y_diff_jacobian

    def load_dok_gradient(self, json_file):
        with open(json_file) as json_file:
            self.dok_gradient = json.load(json_file)

    def gradient(self, y, t, parameters):
        if self.dok_gradient is None:
            raise AssertionError('Dictionary of Keys (DOK) version of matrix needs to be loaded first.' +
                                 'Call method load_dok_gradient.')
        parameters = self._sorting_params(parameters)
        N = self.current_population(y)
        y_gradient = np.zeros(self.num_state, self.num_param)
        # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

        for coord, value in self.dok_gradient.items():
            y_gradient[eval(coord)] = eval(value)
        return y_gradient

    def load_dok_grad_jacobian(self, json_file):
        with open(json_file) as json_file:
            self.dok_grad_jacobian = json.load(json_file)

    def grad_jacobian(self, y, t, parameters):
        if self.dok_grad_jacobian is None:
            raise AssertionError('Dictionary of Keys (DOK) version of matrix needs to be loaded first.' +
                                 'Call method load_dok_grad_jacobian.')
        parameters = self._sorting_params(parameters)
        N = self.current_population(y)
        y_gradient_jacobian = np.zeros(self.num_state, self.num_param)
        # see script deriving_MG_model_jacobian.py for where dok_matrix is derived and saved into json formate.

        for coord, value in self.dok_gradients_jacobian.items():
            y_gradient_jacobian[eval(coord)] = eval(value)
        return y_gradient_jacobian


    #%% Fitting through maximum likelihood stuff below.


    def cost(self, params=None, apply_weighting = True):
        """
        Find the cost/loss given time points and the corresponding
        observations.

        Parameters
        ----------
        params: array like
            input value of the parameters
        apply_weighting: boolean
            If True multiplies array of residuals by weightings, else raw
            residuals are used.

        Returns
        -------
        numeric
            sum of the residuals squared

        Notes
        -----
        Only works with a single target (state)

        See also
        --------
        :meth:`diff_loss`

        """
        if self._lossObj is None:
            raise AssertionError('Loss object not set. Use method "set_loss_object" to set.')
        if self.initial_values is None:
            raise AssertionError('initial variable values object not set.')
        yhat = self._getSolution(params)
        cost = self._lossObj.loss(yhat, apply_weighting = apply_weighting)

        return np.nan_to_num(cost) if cost == np.inf else cost

    def _getSolution(self, params):
        x0 = self._initial_values
        t = self._time_frame
        params = params
        solution = self.integrate(x0, t, params)
        i = self._observation_state_index
        return solution[:, i]

    @property
    def loss_object(self):
        return self._lossObj

    @loss_object.setter
    def loss_object(self, loss_object):
        self._lossObj = loss_object

    @property
    def initial_values(self):
        return self._initial_values

    @initial_values.setter
    def initial_values(self, initial_values):
        self._initial_values = initial_values

    @property
    def initial_time(self):
        return self._initial_time

    @initial_time.setter
    def initial_time(self, initial_time):
        self._initial_time = initial_time

    @property
    def time_frame(self):
        return self._time_frame

    @time_frame.setter
    def time_frame(self, time_frame):
        self._time_frame = time_frame
        self.initial_time = time_frame[0]

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, observations):
        self._observations = observations

    @property
    def observation_state_index(self):
        return self._observation_state_index

    @observation_state_index.setter
    def observation_state_index(self, observation_state_index):
        error_msg = 'observation_state_index must be a single non-negative interger or collection of them.'
        if not isinstance(observation_state_index,(list, tuple, np.array)):
            if isinstance(observation_state_index, int) and observation_state_index >= 0:
                self._observation_state_index = [observation_state_index]
            else:
                raise TypeError(error_msg)
        elif (all(isinstance(item,int) for item in observation_state_index) and
              all(item>=0 for item in observation_state_index)):
            self._observation_state_index = observation_state_index
        else:
            raise TypeError(error_msg)

    @property
    def targetParam(self):
        self._targetParam


    @targetParam.setter
    def targetParam(self,param):
        if isinstance(param,str):
            param = [param]
        for item in param:
            if item not in self.all_parameters:
                raise AssertionError(str(item) + ' is not in all_parameters list. I.E. ' ', '.join(self.all_parameters))
        self._targetParam = param

    def cost_sensitivity(self, params=None):
        """
        Obtain the gradient given input parameters using forward
        sensitivity method.

        Parameters
        ----------
        params: array like
            input value of the parameters

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient

        Notes
        -----
        It calculates the gradient by calling :meth:`jac`

        """
        sens = self.jacobian_of_loss(params=params)
        i = self._observation_state_index
        diff_loss = self._lossObj.diff_loss(sens[:,i])
        grad = self._sensToGradWithoutIndex(sens, diff_loss)

        return grad

    def jacobian_of_loss(self, params):
        """
        Obtain the Jacobian of the loss function given input parameters
        using forward sensitivity method.

        Parameters
        ----------
        params: array like, optional
            input value of the parameters

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ===========  =======================================================
            key          meaning
            ===========  =======================================================
            'sens'       intermediate values over the original ode and all the
                         sensitivities, by state, parameters
            'resid'      residuals given params
            'diff_loss'  derivative of the loss function
            ===========  =======================================================

        See also
        --------
        :meth:`sensitivity`

        """

        # first we want to find out the number of sensitivities required
        # add them to the initial values
        num_sens =  self.num_state*self.num_param
        init_state_sens = np.append(self.initial_values, np.zeros(num_sens))
        if hasattr(self, 'ode_and_sensitivity_jacobian'):
            solution, output = scipy.integrate.odeint(self.ode_and_sensitivity,
                                                      init_state_sens, self._time_frame, args=params,
                                                      Dfun=self.ode_and_sensitivity_jacobian,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)
        else:
            solution, output = scipy.integrate.odeint(self.ode_and_sensitivity,
                                                      init_state_sens, self._time_frame, args=params,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)

        return solution

    def _sensToGradWithoutIndex(self, sens, diffLoss):
        """
        forward sensitivites to g where g is the gradient.
        Indicies obtained using information defined here
        """
        index_out = self._getTargetParamSensIndex()
        return self.sens_to_grad(sens[:, index_out], diffLoss)

    def sens_to_grad(self, sens, diff_loss):
        """
        Forward sensitivites to the gradient.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities
        diff_loss: array like
            derivative of the loss function

        Returns
        -------
        g: :class:`numpy.ndarray`
            gradient of the loss function
        """
        # the number of states which we will have residuals for
        num_s = len(self._observation_state_index)

        assert isinstance(sens, np.ndarray), "Expecting an np.ndarray"
        n, p = sens.shape
        assert n == len(diff_loss), ("Length of sensitivity must equal to " +
                                     "the derivative of the loss function")

        # Divide through to obtain the number of parameters we are inferring
        num_out = int(p/num_s) # number of out parameters

        sens = np.reshape(sens, (n, num_s, num_out), 'F')
        # For moment we are not giving any weighting to observations.
        # for j in range(num_out):
        #     sens[:, :, j] *= self._weight

        grad = functools.reduce(np.add, map(np.dot, diff_loss, sens)).ravel()

        return grad

    def _getTargetParamSensIndex(self):
        # build the indexes to locate the correct parameters
        index_out = list()
        # locate the target indexes
        index_list = self._getTargetParamIndex()
        if isinstance(self._observation_state_index, list):
            for j in self._observation_state_index:
                for i in index_list:
                    # always ignore the first numState because they are
                    # outputs from the actual ode and not the sensitivities.
                    # Hence the +1
                    index_out.append(j + (i + 1) * self.num_state)
        else:
            # else, happy times!
            for i in index_list:
                index_out.append(self._observation_state_index + (i + 1) * self.num_state)

        return np.sort(np.array(index_out)).tolist()

    def _getTargetParamIndex(self):
        """
        Get the indices of the targeted parameters
        """
        # we assume that all the parameters are targets
        if self._targetParam is None:
            index_list = range(0, len(self.all_parameters))
        else:
            index_list = [self.all_parameters.index(param) for param in self._targetParam]

        return index_list

    def ode_and_sensitivity(self, state_param, t, params, by_state=False):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            Whether the output vector should be arranged by state or by
            parameters. If False, then it means that the vector of output is
            arranged according to looping i,j from Sensitivity_{i,j} with i
            being the state and j the param. This is the preferred way because
            it leds to a block diagonal Jacobian

        Returns
        -------
        :class:`list`
            concatenation of 2 element. First contains the ode, second the
            sensitivity. Both are of type :class:`numpy.ndarray`

        See Also
        --------
        :meth:`.sensitivity`, :meth:`.ode`

        '''

        if len(state_param) == self.num_state:
            raise AssertionError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        # there is no safety checks on this because it is impossible to
        # distinguish what is state and what is sensitivity as they are
        # all numeric value that can take the full range (-\infty,\infty)
        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        out1 = self.ode(state, t, *params)
        out2 = self.sensitivity(sens, t, state, params, by_state)
        return np.append(out1, out2)

    def sensitivity(self, S, t, state, params, by_state=False):
        """
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        S: array like
            Which should be :class:`numpy.ndarray`.
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normally zero or one,
            depending on whether the initial conditions are also variables.
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        by_state: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`

        Notes
        -----
        It is different to :meth:`.eval_ode` and :meth:`.eval_jacobian` in
        that the extra input argument is not a parameter

        See Also
        --------
        :meth:`.sensitivity`

        """

        # jacobian * sensitivities + G
        # where G is the gradient
        J = self.jacobian(state, t, *params)
        G = self.gradient(state, t, *params)
        A = np.dot(J, S) + G

        if by_state:
            return np.reshape(A, self.num_state*self.num_param)
        else:
            if not hasattr(self, '_SAUtil'):
                self._set_shape_and_adjust_util()
            return self._SAUtil.matToVecSens(A)

    def _set_shape_and_adjust_util(self):
        import pygom
        self._SAUtil = pygom.ode_utils.shapeAdjust(self.num_state, self.num_param)
    
    def ode_and_sensitivity_jacobian(self, state_param, t, params, by_state=False):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.ode_and_sensitivity`

        '''

        if len(state_param) == self.num_state:
            raise AssertionError("Expecting both the state and the sensitivities")
        else:
            state = state_param[0:self.num_state]

        # now we start the computation
        J = self.jacobian(state, t, *params)
        # create the block diagonal Jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters

        # Note that none of the ode integrator in scipy allow a sparse Jacobian
        # matrix.  All of them accept a banded matrix in packed format but not
        # an actual sparse, or specifying the number of bands.
        outJ = np.kron(np.eye(self.num_param), J)
        # Jacobian of the gradient
        GJ = self.grad_jacobian(state, t, *params)
        # and now we add the gradient
        sensJacobianOfState = GJ + self.sens_jacobian_state(state_param, t, params)

        if by_state:
            arrangeVector = np.zeros(self.num_state * self.num_param)
            k = 0
            for j in range(0, self.num_param):
                for i in range(0, self.num_state):
                    if i == 0:
                        arrangeVector[k] = (i*self.num_state) + j
                    else:
                        arrangeVector[k] = (i*(self.num_state - 1)) + j
                    k += 1

            outJ = outJ[np.array(arrangeVector,int),:]
            idx = np.array(arrangeVector, int)
            sensJacobianOfState = sensJacobianOfState[idx,:]
        # The Jacobian of the ode, then the sensitivities w.r.t state and
        # the sensitivities. In block form.  Theoretically, only the diagonal
        # blocks are important but we output the full matrix for completeness
        return np.asarray(np.bmat([
            [J, np.zeros((self.num_state, self.num_state*self.num_param))],
            [sensJacobianOfState, outJ]
        ]))

    def sens_jacobian_state(self, state_param, t, params):
        '''
        Evaluate the jacobian of the sensitivity w.r.t. the
        state given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as
            well as the sensitivities, which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state *
            number of parameters x number of state]

        '''

        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        return self.eval_sens_jacobian_state(time=t, state=state, sens=sens, params=params)

    def eval_sens_jacobian_state(self, time=None, state=None, sens=None, params=None):
        '''
        Evaluate the jacobian of the sensitivities w.r.t the states given
        parameters, state and time. An extension of :meth:`.sens_jacobian_state`
        but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.parameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.sens_jacobian_state`

        '''

        nS = self.num_state
        nP = self.num_param

        # dot first, then transpose, then reshape
        # basically, some magic
        # don't ask me what is actually going on here, I did it
        # while having my wizard hat on

        return(np.reshape(self.diff_jacobian(state, time, *params).dot(
            self._SAUtil.vecToMatSens(sens)).transpose(), (nS*nP, nS)))