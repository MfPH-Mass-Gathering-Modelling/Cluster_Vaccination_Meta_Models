"""
Creation:
    Author: Martin Grunnill
    Date: 13/09/2022
Description: Class for Multnomial random draw seeding of infections.
    
"""
from numbers import Number
from numpy.random import multinomial
from pandas import DataFrame
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

    def calculate_weighting(self, proportion, parameters):
        parameters_error = ('parameters argument should be a dictionary,' +
                            ' with keys being strings and values being numbers.')
        if not isinstance(parameters, dict):
            raise TypeError(parameters_error)
        if any(not isinstance(value, Number) for value in parameters.values()):
            raise TypeError(parameters_error)
        if any(not isinstance(key,str) for key in parameters.keys()):
            raise TypeError(parameters_error)

        return {state: proportion*(parameters[outflow]**-1)
                for state, outflow in self.outflows.items()}





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

    def calculate_weighting(self, proportions, parameters):
        for index, (branch_name, branch) in enumerate(self.branches.items()):
            branch_weighting = branch.calculate_weighting(proportions[branch_name],
                                                          parameters)
            if index == 0:
                weighting = branch_weighting
            else:
                for state, weight in branch_weighting.items():
                    if state in weighting:
                        weighting[state] += weight
                    else:
                        weighting[state] = weight

        weighting_total = sum(weighting.values())
        weighting = {key:value/weighting_total for key, value in weighting.items()}
        return weighting

    def seed_infections(self, n, proportions, parameters, size=1):
        if not(isinstance(size,int)) or size <= 0:
            raise TypeError('size must be an int >0.')
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
        weighting = self.calculate_weighting(proportions, parameters)
        pvals = list(weighting.values())
        states = list(weighting.keys())
        draw = multinomial(n=n, pvals=pvals, size=size)
        if size>1:
            draw = DataFrame(draw, columns=states)
        else:
            draw = dict(zip(states ,draw[0]))
        return draw





