"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import numpy as np
import pandas as pd
import pygom
import scipy
import functools


class BaseSingleClusterVacModel:
    core_vaccine_groups = []
    ve_delay_groups = []
    ve_wanning_groups = []
    groups_waiting_for_vaccine_effectiveness = []
    states = []
    dead_states = []
    vaccinable_states = []
    observed_states = []
    infectious_states = []
    symptomatic_states = []

    def __init__(self, starting_population,
                 groups_loss_via_vaccination,
                 ve_dicts):
        self.starting_population = starting_population
        self.vaccine_groups = []
        for vaccine_group in self.core_vaccine_groups:
            if vaccine_group in self.ve_delay_groups:
                self.vaccine_groups.append(vaccine_group + '_delay')
                self.groups_waiting_for_vaccine_effectiveness.append(vaccine_group + '_delay')
            self.vaccine_groups.append(vaccine_group)
            if vaccine_group in self.ve_wanning_groups:
                self.vaccine_groups.append(vaccine_group + '_waned')
        groups_loss_via_vaccination_conv = {}
        for key, value in groups_loss_via_vaccination.items():
            if isinstance(value, pd.Series):
                new_value = value.tolist()
            else:
                new_value = value
            if key in self.ve_wanning_groups:
                new_key = key + '_waned'
            else:
                new_key = key
            groups_loss_via_vaccination_conv[new_key] = new_value
        self.groups_loss_via_vaccination = groups_loss_via_vaccination_conv
        self._sorting_states()
        self._attaching_ve_dicts(ve_dicts)
        self.ode_calls_dict = {}
        self.jacobian_calls_dict = {}
        self.dok_jacobian = None
        self.dok_gradient = None
        self.dok_gradients_jacobian = None
        self._lossObj = None
        self._initial_values = None
        self._initial_time = None
        self._observations = None
        self._observation_state_index = None
        self._targetParam = None
        self._time_frame = None
        self.num_param = len(self.all_parameters)


    def _attaching_ve_dicts(self, ve_dicts):
        for name, var in ve_dicts.items():
            error_msg_end = name + ' should be a dictionary with keys being the same as the names of the vaccine groups and values being a float or int >=0 and <=1.'
            if not isinstance(var, dict):
                raise TypeError(name + ' is not a dictionary. ' + error_msg_end)
            if set(var.keys()) != set(self.vaccine_groups):
                raise ValueError(name + "'s keys are not in the list: " + ', '.join(self.vaccine_groups) +
                                 ". " + error_msg_end)
            if not all([isinstance(item, (float, int)) for item in var.values()]):
                raise TypeError(name + ' values are not floats or ints. ' + error_msg_end)
            if not all([0 <= item <= 1 for item in var.values()]):
                raise ValueError(name + ' values are not >=0 and <=1. ' + error_msg_end)
            setattr(self, name, var)

    def _sorting_states(self):
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states]
        self.all_states_index = {}
        self.state_index = {}
        self.infectious_symptomatic_index = {}
        self.infectious_asymptomatic_index = {}
        self.vaccinable_states_index ={}
        self.dead_states_index = {}
        # populating index dictionaries
        index = 0
        for vaccine_group in self.vaccine_groups:
            self.dead_states_index[vaccine_group] = {}
            self.vaccinable_states_index[vaccine_group] = {}
            self.state_index[vaccine_group] = {}
            self.infectious_symptomatic_index[vaccine_group] = {}
            self.infectious_asymptomatic_index[vaccine_group] = {}
            for state in self.states:
                self.all_states_index[state+'_'+vaccine_group]= index
                self.state_index[vaccine_group][state] = index
                if state in self.infectious_and_symptomatic_states:
                    self.infectious_symptomatic_index[vaccine_group][state] = index
                if state in self.infectious_and_asymptomatic_states:
                    self.infectious_asymptomatic_index[vaccine_group][state] = index
                if state in self.dead_states:
                    self.dead_states_index[vaccine_group][state] = index
                if state in self.vaccinable_states:
                    self.vaccinable_states_index[vaccine_group][state] = index
                index += 1

        self.state_index['observed_states'] = {}
        for state in self.observed_states:
            self.all_states_index[state] = index
            self.state_index['observed_states'][state] = index
            index += 1

        self.num_state = index

    def _nesteddictvalues(self, d):
        return [index for sub_d in d.values() for index in sub_d.values()]

    def foi(self, y, beta, asymptomatic_tran_mod):
        """Calculate force of infection (foi).

        Args:
            y (numpy.array): Value of current state variables.
            beta (float or int): Transmission coefficient.
            asymptomatic_tran_mod (float or int): Modification due to asymptomatic/pre-symptomatic state.

        Returns:
            float: Force of infection given state variables.
        """
        infectious_asymptomatic_index = self._nesteddictvalues(self.infectious_asymptomatic_index)
        infectious_symptomatic_index = self._nesteddictvalues(self.infectious_symptomatic_index)
        total_infectous_asymptomatic = y[infectious_asymptomatic_index].sum()
        total_infectous_symptomatic = y[infectious_symptomatic_index].sum()
        total_contactable_population = self.current_population(y)
        foi = (beta * (asymptomatic_tran_mod * total_infectous_asymptomatic +
                       total_infectous_symptomatic) / total_contactable_population)
        return foi

    def current_population(self, y):
        dead_states_index = self._nesteddictvalues(self.dead_states_index)
        return self.starting_population - y[dead_states_index].sum()

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


    def vac_group_transfer(self, y, y_deltas, t,
                           inverse_effective_delay,
                           inverse_waning_immunity,
                           vaccine_group
                           ):
        if vaccine_group not in self.vaccine_groups:
            raise ValueError('vaccine_group "' + vaccine_group + '" not used in model or mispelled in code.' +
                             'Should be one of: "' + '", '.join(self.vaccine_groups) + '".')

        if vaccine_group != self.vaccine_groups[-1]:
            # Unpack y elements relevant to this vaccination group.
            vac_group_states_index = self.state_index[vaccine_group]
            index_of_next_vac_group = self.vaccine_groups.index(vaccine_group) + 1
            next_vaccine_group = self.vaccine_groups[index_of_next_vac_group]
            next_vac_group_states_index = self.state_index[next_vaccine_group]

            # Lets deal with vaccinations first
            ## Then the groups being transfered to the next vaccination group
            if vaccine_group in self.groups_loss_via_vaccination.keys():
                derived_vaccination_rates = self.derived_vaccination_rates[vaccine_group]
                if t not in derived_vaccination_rates.keys():
                    index_of_target_loss = int(t) + 1
                    total_loss_via_vaccination = self.groups_loss_via_vaccination[vaccine_group][index_of_target_loss]
                    if total_loss_via_vaccination == 0:  # No point in calculations if no one is being vaccinated.
                        derived_vaccination_rates[t] = 0
                    else:
                        vaccinable_states_index = list(self.vaccinable_states_index[vaccine_group].values())
                        total_vaccinable = y[vaccinable_states_index].sum()
                        inst_loss_via_vaccine = self.instantaneous_transfer(total_loss_via_vaccination, total_vaccinable, t)
                        derived_vaccination_rates[t] = 1 - np.exp(inst_loss_via_vaccine)
                vaccine_group_transfer = {state: derived_vaccination_rates[t] * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            elif vaccine_group in self.groups_waiting_for_vaccine_effectiveness:
                vaccine_group_transfer = {state: inverse_effective_delay * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            elif vaccine_group in self.ve_wanning_groups:
                vaccine_group_transfer = {state: inverse_waning_immunity * y[vac_group_states_index[state]]
                                          for state in self.vaccinable_states}
            else:
                raise ValueError(
                    'vaccine_group "' + vaccine_group + '" has no method of transfer to next vaccination group ' +
                    'and is not the last vaccination group "' + self.vaccine_groups[-1] + '".')

            for state, transfering_pop in vaccine_group_transfer.items():
                y_deltas[vac_group_states_index[state]] -= transfering_pop
                y_deltas[next_vac_group_states_index[state]] += transfering_pop


    def integrate(self, x0, t, params, full_output=False):
        '''
        A wrapper on top of :mod:`odeint <scipy.integrate.odeint>` using
        :class:`DeterministicOde <pygom.model.DeterministicOde>`.

        Parameters
        ----------
        t: array like
            the time points including initial time
        full_output: bool, optional
            If the additional information from the integration is required

        '''
        self.derived_vaccination_rates = {key: {} for key in self.groups_loss_via_vaccination.keys()}
        # INTEGRATE!!! (shout it out loud, in Dalek voice)
        # determine the number of output we want
        if hasattr(self, 'jacobian'): # May or may not of defined the models Jacobian
            solution, output = scipy.integrate.odeint(self.ode,
                                                      x0, t, args=params,
                                                      Dfun=self.jacobian,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)
        else:
            solution, output = scipy.integrate.odeint(self.ode,
                                                      x0, t, args=params,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)

        if full_output == True:
            # have both
            return solution, output
        else:
            return solution

    ####
    # Fitting through maximum likelihood stuff below.
    #####

    def cost(self, params=None, apply_weighting = True):
        """
        Find the cost/loss given time points and the corresponding
        observations.

        Parameters
        ----------
        params: array like
            input value of the parameters
        apply_weighting: boolean
            If True multiplies array of residuals by weightings, else raw
            residuals are used.

        Returns
        -------
        numeric
            sum of the residuals squared

        Notes
        -----
        Only works with a single target (state)

        See also
        --------
        :meth:`diff_loss`

        """
        if self._lossObj is None:
            raise AssertionError('Loss object not set. Use method "set_loss_object" to set.')
        if self.initial_values is None:
            raise AssertionError('initial variable values object not set.')
        yhat = self._getSolution(params)
        cost = self._lossObj.loss(yhat, apply_weighting = apply_weighting)

        return np.nan_to_num(cost) if cost == np.inf else cost

    def _getSolution(self, params):
        x0 = self._initial_values
        t = self._time_frame
        params = params
        solution = self.integrate(x0, t, params)
        i = self._observation_state_index
        return solution[:, i]

    @property
    def loss_object(self):
        return self._lossObj

    @loss_object.setter
    def loss_object(self, loss_object):
        self._lossObj = loss_object

    @property
    def initial_values(self):
        return self._initial_values

    @initial_values.setter
    def initial_values(self, initial_values):
        self._initial_values = initial_values

    @property
    def initial_time(self):
        return self._initial_time

    @initial_time.setter
    def initial_time(self, initial_time):
        self._initial_time = initial_time

    @property
    def time_frame(self):
        return self._time_frame

    @time_frame.setter
    def time_frame(self, time_frame):
        self._time_frame = time_frame
        self.initial_time = time_frame[0]

    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, observations):
        self._observations = observations

    @property
    def observation_state_index(self):
        return self._observation_state_index

    @observation_state_index.setter
    def observation_state_index(self, observation_state_index):
        error_msg = 'observation_state_index must be a single non-negative interger or collection of them.'
        if not isinstance(observation_state_index,(list, tuple, np.array)):
            if isinstance(observation_state_index, int) and observation_state_index >= 0:
                self._observation_state_index = [observation_state_index]
            else:
                raise TypeError(error_msg)
        elif (all(isinstance(item,int) for item in observation_state_index) and
              all(item>=0 for item in observation_state_index)):
            self._observation_state_index = observation_state_index
        else:
            raise TypeError(error_msg)

    @property
    def targetParam(self):
        self._targetParam


    @targetParam.setter
    def targetParam(self,param):
        if isinstance(param,str):
            param = [param]
        for item in param:
            if item not in self.all_parameters:
                raise AssertionError(str(item) + ' is not in all_parameters list. I.E. ' ', '.join(self.all_parameters))
        self._targetParam = param

    def cost_sensitivity(self, params=None):
        """
        Obtain the gradient given input parameters using forward
        sensitivity method.

        Parameters
        ----------
        params: array like
            input value of the parameters

        Returns
        -------
        grad: :class:`numpy.ndarray`
            array of gradient

        Notes
        -----
        It calculates the gradient by calling :meth:`jac`

        """
        sens = self.jacobian_of_loss(params=params)
        i = self._observation_state_index
        diff_loss = self._lossObj.diff_loss(sens[:,i])
        grad = self._sensToGradWithoutIndex(sens, diff_loss)

        return grad

    def jacobian_of_loss(self, params):
        """
        Obtain the Jacobian of the loss function given input parameters
        using forward sensitivity method.

        Parameters
        ----------
        params: array like, optional
            input value of the parameters

        Returns
        -------
        grad: :class:`numpy.ndarray`
            Jacobian of the objective function
        infodict : dict, only returned if full_output=True
            Dictionary containing additional output information

            ===========  =======================================================
            key          meaning
            ===========  =======================================================
            'sens'       intermediate values over the original ode and all the
                         sensitivities, by state, parameters
            'resid'      residuals given params
            'diff_loss'  derivative of the loss function
            ===========  =======================================================

        See also
        --------
        :meth:`sensitivity`

        """

        # first we want to find out the number of sensitivities required
        # add them to the initial values
        num_sens =  self.num_state*self.num_param
        init_state_sens = np.append(self.initial_values, np.zeros(num_sens))
        if hasattr(self, 'ode_and_sensitivity_jacobian'):
            solution, output = scipy.integrate.odeint(self.ode_and_sensitivity,
                                                      init_state_sens, self._time_frame, args=params,
                                                      Dfun=self.ode_and_sensitivity_jacobian,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)
        else:
            solution, output = scipy.integrate.odeint(self.ode_and_sensitivity,
                                                      init_state_sens, self._time_frame, args=params,
                                                      mu=None, ml=None,
                                                      col_deriv=False,
                                                      mxstep=10000,
                                                      full_output=True)

        return solution

    def _sensToGradWithoutIndex(self, sens, diffLoss):
        """
        forward sensitivites to g where g is the gradient.
        Indicies obtained using information defined here
        """
        index_out = self._getTargetParamSensIndex()
        return self.sens_to_grad(sens[:, index_out], diffLoss)

    def sens_to_grad(self, sens, diff_loss):
        """
        Forward sensitivites to the gradient.

        Parameters
        ----------
        sens: :class:`numpy.ndarray`
            forward sensitivities
        diff_loss: array like
            derivative of the loss function

        Returns
        -------
        g: :class:`numpy.ndarray`
            gradient of the loss function
        """
        # the number of states which we will have residuals for
        num_s = len(self._observation_state_index)

        assert isinstance(sens, np.ndarray), "Expecting an np.ndarray"
        n, p = sens.shape
        assert n == len(diff_loss), ("Length of sensitivity must equal to " +
                                     "the derivative of the loss function")

        # Divide through to obtain the number of parameters we are inferring
        num_out = int(p/num_s) # number of out parameters

        sens = np.reshape(sens, (n, num_s, num_out), 'F')
        # For moment we are not giving any weighting to observations.
        # for j in range(num_out):
        #     sens[:, :, j] *= self._weight

        grad = functools.reduce(np.add, map(np.dot, diff_loss, sens)).ravel()

        return grad

    def _getTargetParamSensIndex(self):
        # build the indexes to locate the correct parameters
        index_out = list()
        # locate the target indexes
        index_list = self._getTargetParamIndex()
        if isinstance(self._observation_state_index, list):
            for j in self._observation_state_index:
                for i in index_list:
                    # always ignore the first numState because they are
                    # outputs from the actual ode and not the sensitivities.
                    # Hence the +1
                    index_out.append(j + (i + 1) * self.num_state)
        else:
            # else, happy times!
            for i in index_list:
                index_out.append(self._observation_state_index + (i + 1) * self.num_state)

        return np.sort(np.array(index_out)).tolist()

    def _getTargetParamIndex(self):
        """
        Get the indices of the targeted parameters
        """
        # we assume that all the parameters are targets
        if self._targetParam is None:
            index_list = range(0, len(self.all_parameters))
        else:
            index_list = [self.all_parameters.index(param) for param in self._targetParam]

        return index_list

    def ode_and_sensitivity(self, state_param, t, params, by_state=False):
        '''
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            Whether the output vector should be arranged by state or by
            parameters. If False, then it means that the vector of output is
            arranged according to looping i,j from Sensitivity_{i,j} with i
            being the state and j the param. This is the preferred way because
            it leds to a block diagonal Jacobian

        Returns
        -------
        :class:`list`
            concatenation of 2 element. First contains the ode, second the
            sensitivity. Both are of type :class:`numpy.ndarray`

        See Also
        --------
        :meth:`.sensitivity`, :meth:`.ode`

        '''

        if len(state_param) == self.num_state:
            raise AssertionError("You have only inputed the initial condition " +
                             "for the states and not the sensitivity")

        # unrolling, assuming that we would always put the state first
        # there is no safety checks on this because it is impossible to
        # distinguish what is state and what is sensitivity as they are
        # all numeric value that can take the full range (-\infty,\infty)
        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        out1 = self.ode(state, t, *params)
        out2 = self.sensitivity(sens, t, state, params, by_state)
        return np.append(out1, out2)

    def sensitivity(self, S, t, state, params, by_state=False):
        """
        Evaluate the sensitivity given state and time

        Parameters
        ----------
        S: array like
            Which should be :class:`numpy.ndarray`.
            The starting sensitivity of size [number of state x number of
            parameters].  Which are normally zero or one,
            depending on whether the initial conditions are also variables.
        t: double
            The current time
        state: array like
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`
        by_state: bool
            how we want the output to be arranged.  Default is True so
            that we have a block diagonal structure

        Returns
        -------
        :class:`numpy.ndarray`

        Notes
        -----
        It is different to :meth:`.eval_ode` and :meth:`.eval_jacobian` in
        that the extra input argument is not a parameter

        See Also
        --------
        :meth:`.sensitivity`

        """

        # jacobian * sensitivities + G
        # where G is the gradient
        J = self.jacobian(state, t, *params)
        G = self.gradient(state, t, *params)
        A = np.dot(J, S) + G

        if by_state:
            return np.reshape(A, self.num_state*self.num_param)
        else:
            if not hasattr(self, '_SAUtil'):
                self._set_shape_and_adjust_util()
            return self._SAUtil.matToVecSens(A)

    def _set_shape_and_adjust_util(self):
        import pygom
        self._SAUtil = pygom.ode_utils.shapeAdjust(self.num_state, self.num_param)
    
    def ode_and_sensitivity_jacobian(self, state_param, t, params, by_state=False):
        '''
        Evaluate the sensitivity given state and time.  Output a block
        diagonal sparse matrix as default.

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as well as the
            sensitivities values all in one.  We assume that the state
            values comes first.
        t: double
            The current time
        by_state: bool
            How the output is arranged, according to the vector of output.
            It can be in terms of state or parameters, where by state means
            that the jacobian is a block diagonal matrix.

        Returns
        -------
        :class:`numpy.ndarray`
            output of a square matrix of size: number of ode + 1 times number
            of parameters

        See Also
        --------
        :meth:`.ode_and_sensitivity`

        '''

        if len(state_param) == self.num_state:
            raise AssertionError("Expecting both the state and the sensitivities")
        else:
            state = state_param[0:self.num_state]

        # now we start the computation
        J = self.jacobian(state, t, *params)
        # create the block diagonal Jacobian, assuming that whoever is
        # calling this function wants it arranges by state-parameters

        # Note that none of the ode integrator in scipy allow a sparse Jacobian
        # matrix.  All of them accept a banded matrix in packed format but not
        # an actual sparse, or specifying the number of bands.
        outJ = np.kron(np.eye(self.num_param), J)
        # Jacobian of the gradient
        GJ = self.grad_jacobian(state, t, *params)
        # and now we add the gradient
        sensJacobianOfState = GJ + self.sens_jacobian_state(state_param, t, params)

        if by_state:
            arrangeVector = np.zeros(self.num_state * self.num_param)
            k = 0
            for j in range(0, self.num_param):
                for i in range(0, self.num_state):
                    if i == 0:
                        arrangeVector[k] = (i*self.num_state) + j
                    else:
                        arrangeVector[k] = (i*(self.num_state - 1)) + j
                    k += 1

            outJ = outJ[np.array(arrangeVector,int),:]
            idx = np.array(arrangeVector, int)
            sensJacobianOfState = sensJacobianOfState[idx,:]
        # The Jacobian of the ode, then the sensitivities w.r.t state and
        # the sensitivities. In block form.  Theoretically, only the diagonal
        # blocks are important but we output the full matrix for completeness
        return np.asarray(np.bmat([
            [J, np.zeros((self.num_state, self.num_state*self.num_param))],
            [sensJacobianOfState, outJ]
        ]))

    def sens_jacobian_state(self, state_param, t, params):
        '''
        Evaluate the jacobian of the sensitivity w.r.t. the
        state given state and time

        Parameters
        ----------
        state_param: array like
            The current numerical value for the states as
            well as the sensitivities, which can be
            :class:`numpy.ndarray` or :class:`list`
        t: double
            The current time

        Returns
        -------
        :class:`numpy.ndarray`
            Matrix of dimension [number of state *
            number of parameters x number of state]

        '''

        state = state_param[0:self.num_state]
        sens = state_param[self.num_state::]

        return self.eval_sens_jacobian_state(time=t, state=state, sens=sens, params=params)

    def eval_sens_jacobian_state(self, time=None, state=None, sens=None, params=None):
        '''
        Evaluate the jacobian of the sensitivities w.r.t the states given
        parameters, state and time. An extension of :meth:`.sens_jacobian_state`
        but now also include the parameters.

        Parameters
        ----------
        parameters: list
            see :meth:`.parameters`
        time: double
            The current time
        state: array list
            The current numerical value for the states which can be
            :class:`numpy.ndarray` or :class:`list`

        Returns
        -------
        :class:`numpy.matrix` or :class:`mpmath.matrix`
            Matrix of dimension [number of state x number of state]

        Notes
        -----
        Name and order of state and time are also different.

        See Also
        --------
        :meth:`.sens_jacobian_state`

        '''

        nS = self.num_state
        nP = self.num_param

        # dot first, then transpose, then reshape
        # basically, some magic
        # don't ask me what is actually going on here, I did it
        # while having my wizard hat on

        return(np.reshape(self.diff_jacobian(state, time, *params).dot(
            self._SAUtil.vecToMatSens(sens)).transpose(), (nS*nP, nS)))