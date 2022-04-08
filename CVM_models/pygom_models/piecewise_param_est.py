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


class PiecewiseParamEstODE(DeterministicOde):
    peicewise_param_estimates = {}
    def __init__(self,
                 state=None,
                 param=None,
                 derived_param=None,
                 transition=None,
                 birth_death=None,
                 ode=None,
                 piecewise_params=None):
        for param, sub_dict in piecewise_params.items():
            targets = sub_dict['targets']
            if isinstance(targets, pd.Series):
                piecewise_params[param]['targets'] = targets.tolist()
            source_states = sub_dict['source states']
            piecewise_params[param]['index of sources'] = [state.index(source_state) for source_state in source_states]
            piecewise_params[param]['estimates'] = {}
        self.piecewise_params = piecewise_params
        self.param_symbols = param
        super().__init__(state, param, derived_param, transition, birth_death, ode)

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
            if key not in self.param_symbols:
                raise ValueError(str(key) + ' is not one of the symbols used in the list of parameters' +
                                 ' used to initialise the instance of this class.')
            if not isinstance(value, Number):
                raise TypeError(str(value) + 'is not a number type.')
        self._fixed_params = fixed_parameters

    def eval_jacobian(self, time, state):
        self.estimate_params_piecewise(time, state)
        return super().eval_jacobian(time=time, state=state)

    def eval_ode(self, time, state):
        self.estimate_params_piecewise(time, state)
        return super().eval_ode(time=time, state=state)

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

    def estimate_params_piecewise(self, time, state):
        piecewise_estimates_at_t = {}
        for param, sub_dict in self.piecewise_params.items():
            target = sub_dict['targets'][int(time) + 1]
            if target == 0:
                estimated_param = 0
            else:
                index_sources = sub_dict['index of sources']
                estimates = sub_dict['estimates']
                total_source = state[index_sources].sum()
                if time not in estimates.keys() or total_source not in estimates[time].keys():
                    instantaneous_transfer = self.instantaneous_transfer(target, total_source, time)
                    estimated_param = 1 - np.exp(instantaneous_transfer)
                    self.piecewise_params[param]['estimates'][time][total_source] = estimated_param
                else:
                    estimated_param = estimates[time][total_source]
            piecewise_estimates_at_t[param] = estimated_param
        self.parameters = {**self.fixed_params, **piecewise_estimates_at_t}