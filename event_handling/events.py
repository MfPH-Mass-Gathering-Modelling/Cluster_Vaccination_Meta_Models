"""
Creation:
    Author: Martin Grunnill
    Date: 20/09/2022
Description: Events for use in class EventQueue.

"""
from numbers import Number
from warnings import warn
import numpy as np
from pygom.model.base_ode_model import BaseOdeModel

class BaseEvent:

    def __init__(self, name):
        if not isinstance(name,str):
            raise TypeError('A name given to an event should be a string.')
        self.name = name


    def process(self):
        pass


class ValueFactorProportionChangeEvent(BaseEvent):
    def __init__(self, name, value=None, factor=None, proportion=None):
        self._value = None
        self._factor = None
        self._proportion = None
        error_msg = 'Only one out of factor, value and proportion can be given.'
        if factor is not None and value is not None and proportion is not None:
            raise AssertionError(error_msg)
        if factor is not None and value is not None:
            raise AssertionError(error_msg)
        if factor is not None and proportion is not None:
            raise AssertionError(error_msg)
        if value is not None and proportion is not None:
            raise AssertionError(error_msg)
        if factor is not None:
            self.factor = factor
        if value is not None:
            self.value = value
        if proportion is not None:
            self.proportion = proportion
        super().__init__(name=name)

    @property
    def proportion(self):
        return self._proportion

    @proportion.setter
    def proportion(self, proportion):
        if not isinstance(proportion, Number):
            raise TypeError('Value for proportion must be a numeric type.')
        if proportion <= 0:
            raise ValueError('Value of ' + str(proportion) + ' entered for proportion must be greater than 0.' +
                             'To turn event to a nullevent (an event that transfers nothing use method "make_event_a_nullevent".')
        if proportion > 1:
            raise ValueError('Value of ' + str(proportion) +
                             ' entered for proportion must be less than or equal to 1.')
        if self.value is not None:
            warn(self.name + ' was set to value, it will now be set to a proportion.')
            self._value = None
        if self.factor is not None:
            warn(self.name + ' was set to factor, it will now be set to a proportion.')
            self._factor = None
        self._proportion = proportion
    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, factor):
        if not isinstance(factor, Number):
            raise TypeError('Value for factor must be a numeric type.')
        if self.value is not None:
            warn(self.name + ' was set to value, it will now be set to a factor.')
            self._value = None
        if self.proportion is not None:
            warn(self.name + ' was set to proportion, it will now be set to a factor.')
            self._proportion = None
        self._factor = factor

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not isinstance(value, Number):
            raise TypeError('Value for value must be a numeric type.')
        if self.factor is not None:
            warn(self.name + ' was set to factor, it will now be set to a value.')
            self._factor = None
        if self.proportion is not None:
            warn(self.name + ' was set to proportion, it will now be set to a value.')
            self._proportion = None
        self._value = value

    def make_event_a_nullevent(self):
        self._value = None
        self._factor = None
        self._proportion = None


class TransferEvent(ValueFactorProportionChangeEvent):
    def __init__(self, name, value=None, factor=None,  proportion=None,
                 from_index=None, to_index=None):
        super().__init__(name=name, value=value, factor=factor, proportion=proportion)
        if from_index is None and to_index is None:
            raise AssertionError('A container of ints must be given for from_index or to_index or both.')
        if from_index is not None:
            if not all(isinstance(index, int) for index in from_index):
                raise TypeError('All values in from_index must be ints.')
            if from_index == to_index:
                raise AssertionError('from_index and to_index must not be equivelant.')
            self.number_of_elements = len(from_index)
        if to_index is not None:
            if not all(isinstance(index, int) for index in from_index):
                raise TypeError('All values in to_index must be ints.')
            self.number_of_elements = len(from_index)
        if from_index is not None and to_index is not None and len(from_index)!=len(to_index):
            raise AssertionError('If both from_index and to_index are given they must be of the same length.')
        self.from_index = from_index
        self.to_index = to_index

    def process(self, solution_at_t, time, return_total_effected=True):
        if self.value is None and\
                self.factor is None and\
                self.proportion is None:
            pass
        else:
            if self.value is not None:
                transfers = np.repeat(self.value, self.number_of_elements)
                if self.from_index is not None:
                    less_than_array = solution_at_t < transfers
                    if any(less_than_array):
                        warn('The total in one or more states was less than default value being deducted'+
                             ','+str(self.value)+', at time ' + str(time) + '.'+
                             ' Removed total population of effected state or states instead.')
                        transfers[less_than_array] = solution_at_t[self.from_index[less_than_array]]
            if self.proportion is not None:
                transfers = solution_at_t[self.from_index] * self.proportion
            if self.factor is not None:
                transfers = solution_at_t[self.from_index] * self.factor
                if self.factor > 1:
                    warn('More people are being transfered than in population.')
            if self.to_index is not None:
                solution_at_t[self.to_index] += transfers
            if self.from_index is not None:
                solution_at_t[self.from_index] -= transfers
            if return_total_effected:
                return transfers.sum()

class ChangeParametersEvent(ValueFactorProportionChangeEvent):
    def __init__(self, name, changing_parameters, value=None, factor=None, proportion=None):
        super().__init__(name=name, value=value, factor=factor, proportion=proportion)
        self.changing_parameters = changing_parameters

    def process(self, model_object, parameters_attribute, parameters):
        if self.value is None and\
                self.factor is None and\
                self.proportion is None:
            pass
        else:
            for parameter in self.changing_parameters:
                if self.value is not None:
                    parameters[parameter] = self.value
                if self.factor is not None:
                    parameters[parameter] *= self.factor
                if self.proportion is not None:
                    parameters[parameter] *= self.proportion

        setattr(model_object, parameters_attribute, parameters)
        return parameters


class ParametersEqualSubPopEvent(BaseEvent):
    def __init__(self, name, changing_parameters, subpopulation_index):
        super().__init__(name=name)
        self.changing_parameters = changing_parameters
        self.subpopulation_index = subpopulation_index

    def process(self, model_object, parameters_attribute, parameters, y):
        value = y[self.subpopulation_index].sum()
        for parameter in self.changing_parameters:
            parameters[parameter] = value

        setattr(model_object, parameters_attribute, parameters)
        return parameters



