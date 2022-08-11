"""
Creation:
    Author: Martin Grunnill
    Date: 10/08/2022
Description: Classes for handling transfer of people between states to another at set timepoints.
    
"""
from collections import OrderedDict, deque
from numbers import Number
import numpy as np
import pandas as pd
from warnings import warn
import copy

class SimModelsWithTranfersAtTs:

    def __init__(self, transfer_info_dict):
        self.master_event_queue = TransferEventQueue(transfer_info_dict)

    def run_simulation(self, y0, func, start_time, end_time, simulation_step=1):
        if not all(time % simulation_step == 0
                   for time in self.event_queue.keys()):
            raise ValueError('All time points for events must be divisible by simulation_step, leaving no remainder.')
        event_queue = copy.deepcopy(self.master_event_queue)
        event_queue.prep_queue_for_sim_time(start_time, end_time)
        tranfers_list = []
        y = []
        current_time = start_time
        solution_at_t = y0
        while event_queue.not_empty():
            next_time, event = event_queue.poptop()
            if current_time != next_time:
                current_t_range = np.arange(current_time, next_time, simulation_step)
            y_over_current_t_range = func(solution_at_t, current_t_range)
            current_time = next_time
            solution_at_t = y_over_current_t_range[-1,:]
            y_over_current_t_range = np.delete(y_over_current_t_range, -1, 0)
            y.append(y_over_current_t_range)
            transfered = event.process(self, solution_at_t, current_time)
            transfers_entry = {'time':current_time,
                               'transfered':transfered,
                               'event':event.name}
            tranfers_list.append(transfers_entry)
        y = np.concatenate(y, axis=0)
        transfers_df = pd.DataFrame(tranfers_list)
        return y, transfers_df
            




class TransferEventQueue:
    def __init__(self, transfer_info_dict):
        unordered_que = {}
        for event_name, event_information in transfer_info_dict.items():
            event_information = copy.deepcopy(event_information) # We don't want to alter the original.
            times = set(event_information['times'])
            del event_information['times']
            event = Event(event_name, **event_information)
            times_already_in_queue = set(unordered_que.keys()) & times
            if times_already_in_queue:
                for time in times_already_in_queue:
                    if isinstance(unordered_que[time], Event):
                        raise warn('Concuring events at time ' + str(time) + '.' +
                                   'Will process events occuring at same time in order of occurance in transfer_info_dict.')
                        unordered_que[time] = deque([unordered_que[time], event])
                    else:
                        unordered_que[time].append(event)
                times -= times_already_in_queue
            unordered_que.update({time: event for time in times})
        self.queue = OrderedDict(sort(unordered_que.items()))

    def poptop(self):
        if self.queue: # OrderedDicts if not empty are seen as True in bool statements.
            next_item = next(iter(self.queue.values()))
            if isinstance(next_item, deque):
                next_time = next(iter(self.queue.keys()))
                next_event = next_item.popleft()
                if not next_item:
                    self.queue.popitem(last=False)
                return next_time, next_event
            else:
                return self.queue.popitem(last=False)
        else:
            raise KeyError("Empty event queue")

    def prep_queue_for_sim_time(self, start_time, end_time):
        first_event_time = next(iter(self.queue.keys()))
        last_event_time = next(reversed(self.queue.keys()))
        if first_event_time < start_time and last_event_time > last_event_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time >= start_time and time <= end_time})
        elif first_event_time < start_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time >= start_time})
        elif last_event_time > last_event_time:
            self.queue = OrderedDict({time: item for time, item
                                      in self.queue.items()
                                      if time <= end_time})
        if end_time in self.queue:
            self.queue[end_time].append(NullEvent)
        else:
            self.queue[end_time] = NullEvent

    def not_empty(self):
        return bool(self.queue)

    def __time_points__(self):
        return len(self.queue)

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


    def process(self, solution_at_t, time, return_total_effected=True):
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
        if return_total_effected:
            return transfers.sum()

class NullEvent:
    def process(self, solution_at_t, time, return_total_effected=True):
        pass



