"""
Creation:
    Author: Martin Grunnill
    Date: 10/08/2022
Description: Classes for handling transfer of people between states to another at set timepoints.
    
"""
from event_handling.events import *
from collections import OrderedDict, deque
from numbers import Number
import numpy as np
import pandas as pd
from warnings import warn
import copy
import inspect

class EventQueue:

    def __init__(self, event_info_dict):
        self.event_queue = _EventQueue(event_info_dict)
        self._master_event_queue = copy.deepcopy(self.event_queue)
        self._events = self.event_queue._events

    def reset_event_queue(self):
        self.event_queue = copy.deepcopy(self._master_event_queue)
        self._events = self.event_queue._events

    def _event_names_checker(self, event_names):
        if event_names == 'all':
            event_names = self.get_event_names()
        else:
            if not pd.api.types.is_list_like(event_names):
                event_names = [event_names]
            if any(not isinstance(item, str) for item in event_names):
                raise TypeError('All event_names entries must be a string.')
            available_event_names = self.get_event_names()
            if any(not item in available_event_names for item in event_names):
                raise TypeError('All event_names entries must be a string.')
        return event_names

    def change_event_proportion(self, event_names, proportion):
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.proportion = proportion

    def change_event_factor(self, event_names, factor):
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.factor = factor

    def change_event_value(self, event_names, value):
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.value = value

    def make_events_nullevents(self, event_names):
        event_names = self._event_names_checker(event_names)
        for event_name in event_names:
            event = self._events[event_name]
            event.make_event_a_nullevent()

    def get_event_names(self):
        return list(self._events.keys())

    def events_at_same_time(self):
        return self.event_queue.events_at_same_time

    def event_names(self):
        return self.event_queue._events.keys()

    def run_simulation(self, model_object, run_attribute, y0, end_time, parameters_attribute, parameters,
                       start_time=0, simulation_step=1,
                       full_output=False, return_param_changes=False,
                       **kwargs_to_pass_to_func):
        if not all(time % simulation_step == 0
                   for time in self.event_queue.times):
            raise ValueError('All time points for events must be divisible by simulation_step, leaving no remainder.')
        setattr(model_object, parameters_attribute, parameters)
        param_changes = {'Starting value, time: '+str(start_time): copy.deepcopy(parameters)}
        sim_event_queue = copy.deepcopy(self.event_queue)
        sim_event_queue.prep_queue_for_sim_time(start_time, end_time)
        tranfers_list = []
        y = []
        current_time = start_time
        solution_at_t = y0
        if full_output:
            run_method = getattr(model_object, run_attribute)
            function_args_inspection = inspect.getfullargspec(run_method)
            full_output_in_func_args = 'full_output' in function_args_inspection.args
            if full_output_in_func_args:
                info_dict = {}
            else:
                warn('Full output unavailable as full_output is not an argument in function given to "func".')

        def func_with_full_output(func, solution_at_t, current_t_range, info_dict, **kwargs_to_pass_to_func):
            y_over_current_t_range, info_sub_dict = func(solution_at_t, current_t_range, full_output=True, **kwargs_to_pass_to_func)
            range_as_list = current_t_range.tolist()
            time_points = (range_as_list[0], range_as_list[-1])
            info_dict[time_points] = info_sub_dict
            return y_over_current_t_range


        while sim_event_queue.not_empty():
            next_time, event = sim_event_queue.poptop()
            if next_time > end_time: # if the next event is after simulation time break out of loop.
                break

            if current_time != next_time:
                # run until current time is next time
                current_t_range = np.arange(current_time, next_time+simulation_step, simulation_step)
                run_method = getattr(model_object, run_attribute)
                if full_output and full_output_in_func_args:
                    y_over_current_t_range = func_with_full_output(run_method, solution_at_t, current_t_range, info_dict, **kwargs_to_pass_to_func)
                else:
                    y_over_current_t_range = run_method(solution_at_t, current_t_range, **kwargs_to_pass_to_func)
                solution_at_t = y_over_current_t_range[-1, :]
                y.append(y_over_current_t_range[:-1, :])
                current_time = next_time
            # then do event
            if isinstance(event, TransferEvent):
                transfered = event.process(solution_at_t, current_time)
                transfers_entry = {'time':current_time,
                                   'transfered':transfered,
                                   'event':event.name}
                tranfers_list.append(transfers_entry)
            elif isinstance(event, ChangeParametersEvent):
                parameters = event.process(model_object, parameters_attribute, parameters)
                param_changes[event.name + ', time: ' + str(current_time)] = copy.deepcopy(parameters)



        if current_time != end_time:
            current_t_range = np.arange(current_time, end_time+simulation_step, simulation_step)
            run_method = getattr(model_object, run_attribute)
            if full_output and full_output_in_func_args:
                y_over_current_t_range = func_with_full_output(run_method, solution_at_t, current_t_range, info_dict, **kwargs_to_pass_to_func)
            else:
                y_over_current_t_range = run_method(solution_at_t, current_t_range, **kwargs_to_pass_to_func)
            y.append(y_over_current_t_range)
        y = np.vstack(y)
        transfers_df = pd.DataFrame(tranfers_list)
        if full_output and full_output_in_func_args:
            if return_param_changes:
                return y, transfers_df, param_changes, info_dict
            else:
                return y, transfers_df, info_dict
        else:
            if return_param_changes:
                return y, transfers_df, param_changes
            else:
                return y, transfers_df


class _EventQueue:
    def __init__(self, transfer_info_dict):
        unordered_que = {}
        self.events_at_same_time = {}
        self._events = {}
        for event_name, event_information in transfer_info_dict.items():
            event_information = copy.deepcopy(event_information) # We don't want to alter the original.
            if isinstance(event_information['times'], Number):
                times = set([event_information['times']])
            else:
                times = set(event_information['times'])
            del event_information['times']
            if event_information['type'] == 'transfer':
                del event_information['type']
                event = TransferEvent(event_name, **event_information)
            elif event_information['type'] == 'change parameter':
                del event_information['type']
                event = ChangeParametersEvent(event_name, **event_information)
            elif event_information['type'] == 'parameter equals subpopulation':
                del event_information['type']
                event = ParametersEqualSubPopEvent(event, **event_information)
            else:
                raise ValueError('Event type not known.')
            self._events[event.name] = event
            times_already_in_queue = set(unordered_que.keys()) & times
            if times_already_in_queue:
                for time in times_already_in_queue:
                    if isinstance(unordered_que[time], BaseEvent):
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

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return f"Queue({self.data.items()})"
