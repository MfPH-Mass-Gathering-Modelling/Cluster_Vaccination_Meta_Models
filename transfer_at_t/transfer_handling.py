"""
Creation:
    Author: Martin Grunnill
    Date: 10/08/2022
Description: Classes for handling transfer of people between states to another at set timepoints.
    
"""
from collections import OrderedDict
from numbers import Number
import numpy as np
from warnings import warn

class SimModelsWithTranfersAtTs:


    def __init__(self, sim_function, transfer_indexes, transfer_info_dict, simulation_step=1):
        self.event_queue = TransferEventQueue(transfer_indexes, transfer_info_dict, simulation_step)
        self.sim_function = sim_function


class TransferEventQueue:
    def __init__(self, transfer_indexes, transfer_info_dict, simulation_step):
        self.data = OrderedDict()

        if initial_data is not None:
            self.data.update(initial_data)
        if kwargs:
            self.data.update(kwargs)

    def enqueue(self, item):
        key, value = item
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value

    def poptop(self):
        try:
            return self.data.popitem(last=False)
        except KeyError:
            print("Empty queue")

    def popbottom(self):
        try:
            return self.data.popitem(last=True)
        except KeyError:
            print("Empty queue")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Queue({self.data.items()})"


class Event:
    def __init__(self, name, proportion=None, amount=None,
                 from_index=None, to_index=None):
        self.name = name
        if proportion is None and amount is None:
            raise AssertionError('Proportion or amount argument must be give.')
        if proportion is not None:
            if amount is not None:
                raise AssertionError('Either proportion or amount argument must be give not both')
            if not isinstance(proportion,Number):
                raise TypeError('Value for proportion must be a numeric type.')
            if proportion <= 0:
                raise ValueError('Value of ' + str(proportion) + ' entered for proportion must be greater than 0.')
            if proportion > 1:
                raise ValueError('Value of ' + str(proportion) +
                                 ' entered for proportion must be less than or equal to 1.')
        if amount is not None:
            if not isinstance(amount,Number):
                raise TypeError('Value for amount must be a numeric type.')
            if amount <= 0:
                raise ValueError('Value of ' + str(amount) + ' entered for amount must be greater than 0.')
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
        if to_index is not None and proportion is not None and from_index is None:
            raise AssertionError('A proportion must be taken from somewhere if going to somewhere.'+
                                 'I.E. if a value is given for to_index and proportion a value must be given for from_index.')
        self.amount = amount
        self.proportion = proportion
        self.from_index = from_index
        self.to_index = to_index


    def process(self, solution_at_t, time, output_transfers=False):
        if self.amount is not None:
            transfers = np.repeat(self.amount, self.number_of_elements)
            if self.from_index is not None:
                less_than_array = solution_at_t < transfers
                if any(less_than_array):
                    warn('The total in one or more states was less than default amount being deducted'+
                         ','+str(self.amount)+', at time ' + str(time) + '.'+
                         ' Removed total population of effected state or states instead.')
                    transfers[less_than_array] = solution_at_t[self.from_index[less_than_array]]
        if self.proportion is not None:
            transfers = solution_at_t[self.from_index] * self.proportion
        if self.to_index is not None:
            solution_at_t[self.to_index] += transfers
        if self.from_index is not None:
            solution_at_t[self.from_index] -= transfers
            if self.amount is not None:
                index_negative = np.where(solution_at_t < 0)
                if index
        if output_transfers:
            return transfers





