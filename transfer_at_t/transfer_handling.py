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
import inspect

class TranfersAtTsScafold:

    def __init__(self, transfer_info_dict):
        self.master_event_queue = TransferEventQueue(transfer_info_dict)
        self._events = self.master_event_queue._events

    def change_transfer_event_proportion(self, event_name, proportion):
        event = self._events[event_name]
        event.proportion = proportion

    def change_transfer_event_amount(self, event_name, amount):
        event = self._events[event_name]
        event.amount = amount

    def events_at_same_time(self):
        return self.master_event_queue.events_at_same_time

    def event_names(self):
        return self.master_event_queue._events.keys()

    def run_simulation(self, func, y0,  end_time, start_time=0, simulation_step=1, full_output=False,
                       args_to_pass_to_func={}):
        if not all(time % simulation_step == 0
                   for time in self.master_event_queue.times):
            raise ValueError('All time points for events must be divisible by simulation_step, leaving no remainder.')
        event_queue = copy.deepcopy(self.master_event_queue)
        event_queue.prep_queue_for_sim_time(start_time, end_time)
        tranfers_list = []
        y = []
        current_time = start_time
        solution_at_t = y0
        next_time, event = event_queue.poptop()
        if full_output:
            function_args_inspection = inspect.getfullargspec(func)
            full_output_in_func_args = 'full_output' in function_args_inspection.args
            if full_output_in_func_args:
                info_dict = {}
            else:
                warn('Full output unavailable as full_output is not an argument in function given to "func".')

        def func_with_full_output(func, solution_at_t, current_t_range, info_dict, args_to_pass_to_func):
            y_over_current_t_range, info_sub_dict = func(solution_at_t, current_t_range, full_output=True, **args_to_pass_to_func)
            range_as_list = current_t_range.tolist()
            time_points = (range_as_list[0], range_as_list[-1])
            info_dict[time_points] = info_sub_dict
            return y_over_current_t_range

        while event_queue.not_empty():
            if current_time != next_time:
                current_t_range = np.arange(current_time, next_time+simulation_step, simulation_step)
                if full_output and full_output_in_func_args:
                    y_over_current_t_range = func_with_full_output(func, solution_at_t, current_t_range, info_dict)
                else:
                    y_over_current_t_range = func(solution_at_t, current_t_range, **args_to_pass_to_func)
                solution_at_t = y_over_current_t_range[-1, :]
                y.append(y_over_current_t_range[:-1, :])
            transfered = event.process(solution_at_t, current_time)
            transfers_entry = {'time':current_time,
                               'transfered':transfered,
                               'event':event.name}
            tranfers_list.append(transfers_entry)
            current_time = next_time
            next_time, event = event_queue.poptop()

        if current_time != end_time:
            current_t_range = np.arange(current_time, end_time+simulation_step, simulation_step)
            if full_output and full_output_in_func_args:
                y_over_current_t_range = func_with_full_output(func, solution_at_t, current_t_range, info_dict, args_to_pass_to_func)
            else:
                y_over_current_t_range = func(solution_at_t, current_t_range, **args_to_pass_to_func)
            y.append(y_over_current_t_range)
        y = np.vstack(y)
        transfers_df = pd.DataFrame(tranfers_list)
        if full_output and full_output_in_func_args:
            return y, transfers_df, info_dict
        else:
            return y, transfers_df
            




class TransferEventQueue:
    def __init__(self, transfer_info_dict):
        unordered_que = {}
        self.events_at_same_time = {}
        self._events = {}
        for event_name, event_information in transfer_info_dict.items():
            event_information = copy.deepcopy(event_information) # We don't want to alter the original.
            times = set(event_information['times'])
            del event_information['times']
            event = Event(event_name, **event_information)
            self._events[event.name] = event
            times_already_in_queue = set(unordered_que.keys()) & times
            if times_already_in_queue:
                for time in times_already_in_queue:
                    if isinstance(unordered_que[time], Event):
                        self.events_at_same_time[time] = [unordered_que[time].name, event.name]
                        unordered_que[time] = deque([unordered_que[time], event])
                    else:
                        self.events_at_same_time[time].append(event_name)
                        unordered_que[time].append(event)
                times -= times_already_in_queue
            unordered_que.update({time: event for time in times})
        self.queue = OrderedDict(sorted(unordered_que.items()))
        if self.events_at_same_time:
            warn('Concuring events in event queue. To view use method "events_at_same_time".')

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

    def not_empty(self):
        return bool(self.queue)

    @property
    def times(self):
        return self.queue.keys()

    def __time_points__(self):
        return len(self.queue)

    def __repr__(self):
        return f"Queue({self.data.items()})"


class Event:
    def __init__(self, name, proportion=None, amount=None,
                 from_index=None, to_index=None):
        self.name = name
        self._proportion = None
        self._amount = None
        if proportion is None and amount is None:
            raise AssertionError('Proportion or amount argument must be give.')
        if proportion is not None:
            if amount is not None:
                raise AssertionError('Either proportion or amount argument must be give not both')
            self.proportion = proportion
        if amount is not None:
            self.amount = amount
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
        self.from_index = from_index
        self.to_index = to_index

    @property
    def proportion(self):
        return self._proportion

    @proportion.setter
    def proportion(self, proportion):
        if not isinstance(proportion, Number):
            raise TypeError('Value for proportion must be a numeric type.')
        if proportion < 0:
            raise ValueError('Value of ' + str(proportion) + ' entered for proportion must be equal to or greater than 0.')
        if proportion > 1:
            raise ValueError('Value of ' + str(proportion) +
                             ' entered for proportion must be less than or equal to 1.')
        if self.amount is not None:
            warn(self.name + ' was set to transfer an amount, it will now be set to transfer a proportion of those available.')
            self._amount = None
        self._proportion = proportion

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, amount):
        if not isinstance(amount, Number):
            raise TypeError('Value for amount must be a numeric type.')
        if amount < 0:
            raise ValueError('Value of ' + str(amount) + ' entered for amount must be equal to or greater than 0.')
        if self.proportion is not None:
            warn(self.name + ' was set to transfer a proportion of those available, it will now be set to transfer an amount.')
            self._proportion = None
        self._amount = amount

    def process(self, solution_at_t, time, return_total_effected=True):
        if self.amount is not None:
            if self.amount == 0:
                pass
            transfers = np.repeat(self.amount, self.number_of_elements)
            if self.from_index is not None:
                less_than_array = solution_at_t < transfers
                if any(less_than_array):
                    warn('The total in one or more states was less than default amount being deducted'+
                         ','+str(self.amount)+', at time ' + str(time) + '.'+
                         ' Removed total population of effected state or states instead.')
                    transfers[less_than_array] = solution_at_t[self.from_index[less_than_array]]
        if self.proportion is not None:
            if self.proportion == 0:
                pass
            transfers = solution_at_t[self.from_index] * self.proportion
        if self.to_index is not None:
            solution_at_t[self.to_index] += transfers
        if self.from_index is not None:
            solution_at_t[self.from_index] -= transfers
        if return_total_effected:
            return transfers.sum()





