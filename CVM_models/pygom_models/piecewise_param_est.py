"""
Creation:
    Author: Martin Grunnill
    Date: 08/04/2022
Description: PyGOM model class for running deterministic ODE models with piecewise estimation of a parameter(s).
    
"""
from pygom import DeterministicOde
import pandas as pd
import numpy as np
from numbers import Number
import warnings


class PiecewiseParamEstODE(DeterministicOde):
    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 birth_death=None,
                 ode=None):
        self._piecewise_params = None
        self._fixed_params = None
        self.peicewise_param_estimates = {}
        super().__init__(state, param, derived_param, transition, birth_death, ode)

    @property
    def piecwise_params(self):
        return self._piecewise_params
        
    @piecwise_params.setter
    def piecwise_params(self, piecewise_params):
        for param, sub_dict in piecewise_params.items():
            if param not in self._paramDict.keys():
                raise ValueError(str(param) + ' is not one of the symbols used in the list of parameters' +
                                 ' used to initialise the instance of this class. One of ' +
                                 ','.join(self._paramDict.keys())+'.')

            targets = sub_dict['targets']
            if isinstance(targets, pd.Series):
                piecewise_params[param]['targets'] = targets.tolist()
            source_states = sub_dict['source states']
            piecewise_params[param]['index of sources'] = [self.get_state_index(source_state)
                                                           for source_state in source_states]
            piecewise_params[param]['estimates'] = {}
        self._piecewise_params = piecewise_params


    @property
    def fixed_parameters(self):
        """
        Returns
        -------
        dict
            A dictionary of fixed parameter.
            {string : numeric}

        """
        return self._fixed_params

    @fixed_parameters.setter
    def fixed_parameters(self, fixed_parameters):
        if not isinstance(fixed_parameters,dict):
            raise TypeError('fixed_parameters should be a dictionary.')
        for key, value in fixed_parameters.items():
            if key not in self._paramDict.keys():
                raise ValueError(str(key) + ' is not one of the symbols used in the list of parameters' +
                                 ' used to initialise the instance of this class. One of ' +
                                 ','.join(self._paramDict.keys())+'.')
            if not isinstance(value, Number):
                raise TypeError(str(value) + 'is not a number type.')
        self._fixed_params = fixed_parameters

    def eval_jacobian(self, time, state):
        if self._Jacobian is None or self._hasNewTransition.Jacobian:
            # self.get_ode_eqn()
            self.get_jacobian_eqn()
        paramValue = self.estimate_params_piecewise(time, state)
        eval_param = self._getEvalParam(state=state,time=time, parameters=paramValue)
        return self._JacobianCompile(eval_param)

    def eval_ode(self, time, state):
        if self._ode is None or self._hasNewTransition.ode:
            self.get_ode_eqn()
        paramValue = self.estimate_params_piecewise(time, state)
        eval_param = self._getEvalParam(state=state,time=time, parameters=paramValue)
        return self._odeCompile(eval_param)

    def entire_population_tranfered(self):
        return self.lack_of_people_to_transition

    def instantaneous_transfer(self,
                               population_transitioning,
                               population,
                               t=None,
                               source_states=None):
        if population_transitioning > population:
            if self._not_enough_to_transition_raises_warning:
                if not self.lack_of_people_to_transition:
                    error_msg = ("Population transitioning is greater than available population." +
                                 'Therefore transfering entire population'+
                                 " Use method entire_population_tranfered to view details of when this happened.")
                    warnings.warn(error_msg)
                lack_of_people_to_transition_t = {'population transitioning': population_transitioning,
                                                  'available population': population}
                if source_states is not None:
                    lack_of_people_to_transition_t.update({'source states': source_states})
                if t is not None:
                    lack_of_people_to_transition_t.update({'time': t})
                self.lack_of_people_to_transition.append(lack_of_people_to_transition_t)
                population_transitioning = 0
            else:
                error_msg = "population_transitioning (" + str(
                    population_transitioning) + ") is greater than population (" + str(population)
                if source_states is None:
                    error_msg += ')'
                else:
                    error_msg += ', between states ' + ','.join(source_states) + ')'
                if t is None:
                    error_msg += '.'
                else:
                    error_msg += ', at time ' + str(t) + ').'
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

    def estimate_params_piecewise(self, time, state):
        if self._piecewise_params is None:
            AssertionError('piecewise_params dictionary has not been set.' +
                           ' To set call "the instance of this class".piecewise_params = "your chosen piecewise_params dictionary".')
        if self._fixed_params is None:
            AssertionError('fixed_params dictionary has not been set.' +
                           ' To set call "the instance of this class".fixed_params = "your chosen fixed_params dictionary."')
        piecewise_estimates_at_t = {}
        for param, sub_dict in self._piecewise_params.items():
            target = sub_dict['targets'][int(time) + 1]
            if target == 0:
                estimated_param = 0
            else:
                index_sources = sub_dict['index of sources']
                source_states = sub_dict['source states']
                estimates = sub_dict['estimates']
                total_source = np.take(state, index_sources, axis=0).sum()
                if time not in estimates.keys() or total_source not in estimates[time].keys():
                    if time not in estimates.keys():
                        self._piecewise_params[param]['estimates'][time] = {}
                    instantaneous_transfer = self.instantaneous_transfer(target, total_source, time, source_states)
                    estimated_param = 1 - np.exp(instantaneous_transfer)
                    self._piecewise_params[param]['estimates'][time][total_source] = estimated_param
                else:
                    estimated_param = estimates[time][total_source]
            piecewise_estimates_at_t[param] = estimated_param

        temp_params_dict = {**self._fixed_params, **piecewise_estimates_at_t}
        paramValue = [0]*self.num_param
        for key, val in temp_params_dict.items():
            index = self.get_param_index(key)
            paramValue[index] = val

        return paramValue

    def integrate(self, t, full_output=False, not_enough_to_transition_raises_warning=True):
        '''
        Integrate over a range of t when t is an array and a output at time t

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        full_output: bool
            if we want additional information
        not_enough_to_transition_raises_warning : bool
            If true during piecewise if the target is greater than the source
             population the total population is transitioned. If false an
             error is raised.
        
        '''
        self._not_enough_to_transition_raises_warning = not_enough_to_transition_raises_warning
        if not_enough_to_transition_raises_warning:
            self.lack_of_people_to_transition =[]
        return super().integrate(t, full_output)
    
    def integrate2(self, t, full_output=False, method=None,
                   not_enough_to_transition_raises_warning=True):
        '''
        Integrate over a range of t when t is an array and a output
        at time t.  Select a suitable method to integrate when
        method is None.

        Parameters
        ----------
        t: array like
            the range of time points which we want to see the result of
        full_output: bool
            if we want additional information
        method: str, optional
            the integration method.  All those available in
            :class:`ode <scipy.integrate.ode>` are allowed with 'vode'
            and 'ivode' representing the non-stiff and stiff version
            respectively.  Defaults to None, which tries to choose the
            integration method via eigenvalue analysis (only one) using
            the initial conditions
        not_enough_to_transition_raises_warning : bool
            If true during piecewise if the target is greater than the source
             population the total population is transitioned. If false an
             error is raised.
        '''
        self._not_enough_to_transition_raises_warning = not_enough_to_transition_raises_warning
        if not_enough_to_transition_raises_warning:
            self.lack_of_people_to_transition =[]
        return super().integrate2(t, full_output, method)

    def _getEvalParam(self, state, time, parameters):
        if state is None or time is None:
            raise AssertionError("Have to input both state and time")

        if parameters is not None:
            paramValue = parameters
        elif self._parameters is None:
            if self.num_param == 0:
                pass
            else:
                raise AssertionError("Have not set the parameters yet or given them.")
            paramValue = self.paramValue

        if isinstance(state, list):
            eval_param = state + [time]
        elif hasattr(state, '__iter__'):
            eval_param = list(state) + [time]
        else:
            eval_param = [state] + [time]

        return eval_param + paramValue