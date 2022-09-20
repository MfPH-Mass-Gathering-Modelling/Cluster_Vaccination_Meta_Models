"""
Creation:
    Author: Martin Grunnill
    Date: 20/09/2022
Description: Events for use in class EventQueue.

"""
from numbers import Number
from warnings import warn
import numpy as np

class BaseEvent:

    def __init__(self, name, factor=None, value=None):
        self._value = None
        self._factor = None
        if factor is not None and value is not None:
            raise AssertionError('Proportion or value argument can not both be given.')
        if factor is not None:
            self.factor = factor
        if value is not None:
            self.value = value
        self.name = name
    
    @property
    def factor(self):
        return self._factor

    @factor.setter
    def factor(self, factor):
        if not isinstance(factor, Number):
            raise TypeError('Value for factor must be a numeric type.')
        if factor < 0:
            raise ValueError('Value of ' + str(factor) + ' entered for factor must be greater than or equal to 0.')
        if self.value is not None:
            warn(self.name + ' was set to value, it will now be set to a factor.')
            self._value = None
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
        self._value = value

    def make_event_a_nullevent(self):
        self._value = None
        self._factor = None

    def process(self):
        pass


class TransferEvent(BaseEvent):
    def __init__(self, name, factor=None, value=None,
                 from_index=None, to_index=None):
        super().__init__(name, factor, value)
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
        if self.value is None and self.factor is None:
            super().process()
        else:
            if self.value is not None:
                transfers = np.repeat(self.value, self.number_of_elements)
                if self.from_index is not None:
                    less_than_array = solution_at_t < transfers
                    if any(less_than_array):
                        warn('The total in one or more states was less than default amount being deducted'+
                             ','+str(self.value)+', at time ' + str(time) + '.'+
                             ' Removed total population of effected state or states instead.')
                        transfers[less_than_array] = solution_at_t[self.from_index[less_than_array]]
            if self.factor is not None:
                transfers = solution_at_t[self.from_index] * self.factor
            if self.to_index is not None:
                solution_at_t[self.to_index] += transfers
            if self.from_index is not None:
                solution_at_t[self.from_index] -= transfers
            if return_total_effected:
                return transfers.sum()

class ChangeParametersEvent(BaseEvent):
    def __init__(self, name, parameters, factor=None, value=None):
        super().__init__(name, factor, value)
        self.parameters = parameters

    def process(self, model_object, args, arg_attribute):
        if self.value is None and self.factor is None:
            super().process()
        else:
            for parameter in self.parameters:
                if self.factor is None:
                    args[parameter] = self.value
                else:
                    args[parameter] = args[parameter]*self.factor
            setattr(model_object, arg_attribute, args)






