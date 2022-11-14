"""
Creation:
    Author: Martin Grunnill
    Date: 13/09/2022
Description: Class for Multnomial random draw seeding of infections.
    
"""
from numbers import Number
from numpy.random import multinomial
import math

class InfectionBranch:
    def __init__(self, branch_name, outflows):
        if not isinstance(branch_name,str):
            raise TypeError('branch_name argument should be a string.')
        self.name = branch_name
        outflows_err_msg = ('outflows argument should be a dictionary,'+
                            ' with keys being strings or integers and values being string.')
        if not isinstance(outflows,dict):
            raise TypeError(outflows_err_msg)
        if any(not isinstance(key,(int,str)) for key in outflows.keys()):
            raise TypeError(outflows_err_msg)
        if any(not isinstance(value,str) for value in outflows.values()):
            raise TypeError(outflows_err_msg)
        self.outflows = outflows

    def calculate_weighting(self, parameters):
        parameters_error = ('parameters argument should be a dictionary,' +
                            ' with keys being strings and values being numbers.')
        if not isinstance(parameters, dict):
            raise TypeError(parameters_error)
        if any(not isinstance(value, Number) for value in parameters.values()):
            raise TypeError(parameters_error)
        if any(not isinstance(key,str) for key in parameters.keys()):
            raise TypeError(parameters_error)

        weightings = {}
        total = 0
        for state, outflow in self.outflows.items():
            weighting = parameters[outflow] ** -1
            weightings[state] = weighting
            total += weighting

        noramlised_weightings = {state: weight/total for state, weight in weightings.items()}

        return noramlised_weightings

    def seed_infections(self, n, parameters):
        weighting = self.calculate_weighting(parameters)
        pvals = list(weighting.values())
        states = list(weighting.keys())
        draw = multinomial(n=n, pvals=pvals, size=1)
        draw = draw[0]
        draw_dict = {state: draw[index] for index, state in enumerate(states)}
        return draw_dict




class MultnomialSeeder:

    def __init__(self, branch_info):
        if not isinstance(branch_info,dict):
            raise TypeError('branch_info should be a dictionary.')
        self.branches = {}
        self.parameters = set()
        for branch_name, outflows in branch_info.items():
            self.parameters.update(list(outflows.values()))
            if not isinstance(branch_info, dict):
                raise TypeError('branch_info should be a dictionary of dictionaries.')
            self.branches[branch_name] = InfectionBranch(branch_name, outflows)
    def seed_branches(self, n, proportions):
        pvals = list(proportions.values())
        branches = list(proportions.keys())
        draw = multinomial(n=n, pvals=pvals, size=1)
        draw = draw[0]
        draw_dict = {branch: draw[index] for index, branch in enumerate(branches)}
        return draw_dict
    def seed_infections(self, n, proportions, parameters):
        prob_error = ', all proportion argument should be a number <=1 and >=0.'
        for key, value in proportions.items():
            if not isinstance(value, Number):
                raise TypeError(key+' not a Number type'+ prob_error)
            if value > 1 or value < 0:
                raise ValueError(key+' is of value '+ str(value) + prob_error)
        proportions_total = sum(proportions.values())
        if not math.isclose(1, proportions_total, abs_tol=0.000001):
            raise ValueError('The sum of dictionary values in proportions should equal 1, it is equal to ' +
                             str(proportions_total)+'.')
        branch_draw = self.seed_branches(n, proportions)
        infections_draw = {}
        for branch_name, branch_seed in branch_draw.items():
            branch = self.branches[branch_name]
            branch_infection_draw = branch.seed_infections(branch_seed, parameters)
            states_already_drawn = set(infections_draw.keys()).union(set(branch_infection_draw.keys()))
            updated_infection_draws = {state: branch_infection_draw.get(state, 0) + infections_draw .get(state, 0)
                                       for state in states_already_drawn}
            infections_draw = updated_infection_draws
        return infections_draw





