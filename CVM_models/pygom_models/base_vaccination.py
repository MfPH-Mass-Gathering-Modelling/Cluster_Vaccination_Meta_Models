"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import copy
from pygom import TransitionType, Transition, DeterministicOde, SimulateOde
from CVM_models.pygom_models.piecewise_param_est import PiecewiseParamEstODE

class BaseMultiClusterVacConstructor:
    states = ['S']
    observed_states = []
    infectious_states = []
    symptomatic_states = []
    isolating_states = []
    dead_states = []
    universal_params = []
    vaccine_specific_params = []
    cluster_specific_params = ['N', 'kappa_D']
    vaccine_and_cluster_specific_params = []


    def __init__(self, group_structure):
        self.stoc_model = None
        self.det_model = None
        self.piecewise_param_est_model = None
        self.all_parameters = None
        self.beta_list = []
        self.lambda_dict = {}
        self.transmission_combos = []
        self.transitions = []
        self.bd_list = []
        self.derived_params = []
        self.gen_group_structure(group_structure)
        self.vaccine_specific_params_dict = {vaccine_specific_param:
                                                 [vaccine_specific_param +'_'+ vaccine_group
                                                  for vaccine_group in self.vaccine_groups]
                                             for vaccine_specific_param in
                                             self.vaccine_specific_params}
        self.cluster_specific_params_dict = {cluster_specific_param:
                                                 [cluster_specific_param +'_'+cluster for cluster in self.clusters]
                                              for cluster_specific_param in
                                              self.cluster_specific_params
                                             }
        self.vaccine_and_cluster_specific_params_dict = {vaccine_and_cluster_specific_param: []
                                                          for vaccine_and_cluster_specific_param in
                                                          self.vaccine_and_cluster_specific_params}
        self.states_dict = {state: [] for state in self.states}
        # Setting up clusters
        self.cluster_dead_population = {}
        self.cluste_vaccine_group_state_index = {}
        index = 0
        self.all_states = []
        for cluster_i in self.clusters:
            dead_pop = []
            self.cluste_vaccine_group_state_index[cluster_i] = {}
            for vaccine_group in self.vaccine_groups:
                self.cluste_vaccine_group_state_index[cluster_i][vaccine_group] = {}
                for state in self.states:
                    self.cluste_vaccine_group_state_index[cluster_i][vaccine_group][state] = index
                    index += 1
                    state_in_cluster = state + "_" + cluster_i + "_" + vaccine_group
                    self.states_dict[state].append(state_in_cluster)
                    self.all_states.append(state_in_cluster)
                    if state in self.dead_states:
                        dead_pop.append(state_in_cluster)

            self.cluster_dead_population[cluster_i] = '+'.join(dead_pop)
        self.observation_state_index = {}
        for state in self.observed_states:
            self.observation_state_index[state] = index
            self.all_states.append(state)
            index += 1
        self.num_states = index
        # setting up forces of infection if you have infectious states
        if len(self.infectious_states) > 0:
            self.append_semetric_transmission_from_list(self.clusters)
            self.derived_params = [('lambda_' + cluster_i, '+'.join(self.lambda_dict[cluster_i]))
                                   for cluster_i in self.clusters]

    def _attach_Params(self):
        self.all_parameters = set(self.universal_params + self.beta_list +
                                  list(self.group_transition_params_dict.keys()))
        dictionary_list = [
            self.vaccine_specific_params_dict,
            self.cluster_specific_params_dict,
            self.vaccine_and_cluster_specific_params_dict
        ]
        for specific_params_dict in dictionary_list:
            for list_item in specific_params_dict.values():
                self.all_parameters.update(list_item)

    def gen_group_structure(self, group_structure):
        self.group_transition_params_dict = {}
        if isinstance(group_structure, dict):
            self.vaccine_groups = group_structure['vaccine groups']
            self.clusters = group_structure['clusters']
        elif isinstance(group_structure, list):
            self.vaccine_groups = []
            self.clusters = []
            for group_transfer in group_structure:
                cluster = group_transfer['from_cluster']
                if cluster not in self.clusters:
                    self.clusters.append(cluster)
                vaccine_group = group_transfer['from_vaccine_group']
                if vaccine_group not in self.vaccine_groups:
                    self.vaccine_groups.append(vaccine_group)
                to_cluster = group_transfer['to_cluster']
                if to_cluster not in self.clusters:
                    self.clusters.append(to_cluster)
                to_vaccine_group = group_transfer['to_vaccine_group']
                if to_vaccine_group not in self.vaccine_groups:
                    self.vaccine_groups.append(to_vaccine_group)
                states = group_transfer['states']
                if states == 'all':
                    states = self.states
                else:
                    for state in states:
                        self._check_string_in_list_strings(state, 'states')
                parameter = group_transfer['parameter']
                if parameter not in self.group_transition_params_dict:
                    self.group_transition_params_dict[parameter] = []
                entry = {key: value for key, value in
                         group_transfer.items()
                         if key != 'parameter'}
                self.group_transition_params_dict[parameter].append(entry)
                for state in states:
                    origin = state +'_' + cluster + '_' + vaccine_group
                    destination = state + '_' + to_cluster + '_' + to_vaccine_group
                    self.transitions.append(Transition(origin=origin,
                                                       destination=destination,
                                                       equation=parameter + '*' + origin,
                                                       transition_type=TransitionType.T))


    def _check_string_in_list_strings(self, string, list_strings):
        if not isinstance(string,str):
            raise TypeError(str(string) +' should be of type string.')

        check_list = eval('self.' + list_strings)
        if string not in check_list:
            raise ValueError(string + ' is not one of the predefined model ' + list_strings + ': ' +
                             ','.join(check_list[:-1]) + ' and ' + check_list[:-1] +'.')

    # functions for appending to dictionary of lambdas and list of betas
    def append_transmission(self, cluster_i, cluster_j):
        """Append cluster_i cluster_j transmssion term to beta_list and self.lambda_dict.

        Args:
            cluster_i (string): [description]
            cluster_j (string): [description]
        """
        transmission_combo = (cluster_i, cluster_j)
        if transmission_combo not in self.transmission_combos:
            self.transmission_combos.append(transmission_combo)
            kappa_D = 'kappa_D_' + cluster_j
            contributers = []
            for infectous_state in self.infectious_states:
                modifier = ''
                if infectous_state not in self.symptomatic_states:
                    modifier += 'theta*'
                if infectous_state in self.isolating_states:
                    modifier += kappa_D+'*'
                infectous_state = modifier + infectous_state
                contributers.append(infectous_state)
            temp_lambda = [infectous_state + '_' + cluster_j + '_' + vaccine_group
                           for infectous_state in contributers
                           for vaccine_group in self.vaccine_groups]
            beta = 'beta_' + cluster_i + '_' + cluster_j
            self.beta_list.append(beta)
            if cluster_i not in self.lambda_dict:
                self.lambda_dict[cluster_i] = []
            n_j = 'N_' + cluster_j
            cluster_lambda = '(' + '+'.join(temp_lambda) + ')*' + beta
            if len(self.dead_states) > 0:
                cluster_dead_population = "("+self.cluster_dead_population[cluster_j]+")"
                cluster_lambda += '/(' +n_j + '-'+ cluster_dead_population + ')'
            else:
                cluster_lambda += '/' + n_j
            self.lambda_dict[cluster_i].append(cluster_lambda)

    def append_intra_transmission(self, cluster_i):
        """Append cluster_i cluster_i transmssion term to beta_list and self.lambda_dict.

        Args:
            beta_list (list): [description]
            self.lambda_dict (dict): [description]
            cluster_i (string): [description]
        """
        self.append_transmission(cluster_i, cluster_i)

    def append_semetric_transmission(self, cluster_i, cluster_j):
        self.append_transmission(cluster_i, cluster_j)
        self.append_transmission(cluster_j, cluster_i)

    def append_semetric_transmission_from_list(self, cluster_list, include_intra=True):

        """Appends semetric tranmission to beta_list and self.lambda_dict for all clusters within a list.

        Args:
            beta_list (list): [description]
            self.lambda_dict (dict): [description]
            cluster_list (list of strings): List of cluster to create forces of infection between.
            include_intra (bool): Include generation of transmission within cluster.
        """
        for cluster_i in cluster_list:
            for cluster_j in cluster_list:
                if cluster_i != cluster_j:
                    self.append_semetric_transmission(cluster_i, cluster_j)
                elif include_intra:
                    self.append_intra_transmission(cluster_i)


    def generate_model(self, variety='deterministic'):
        """Generate pygom based model.
        """
        if self.all_parameters is None:
            self._attach_Params()
        if variety=='stochastic':
            if self.stoc_model is None:
                self.stoc_model = SimulateOde(self.all_states,
                                              self.all_parameters,
                                              derived_param=self.derived_params,
                                              transition=self.transitions,
                                              birth_death=self.bd_list)
            model = copy.deepcopy(self.stoc_model)
        elif variety=='deterministic':
            if self.det_model is None:
                self.det_model = DeterministicOde(self.all_states,
                                                  self.all_parameters,
                                                  derived_param=self.derived_params,
                                                  transition=self.transitions,
                                                  birth_death=self.bd_list)
            model = copy.deepcopy(self.det_model)
        elif variety=='piecewise parameter estimation':
            if self.piecewise_param_est_model is None:
                self.piecewise_param_est_model = PiecewiseParamEstODE(self.all_states,
                                                                      self.all_parameters,
                                                                      derived_param=self.derived_params,
                                                                      transition=self.transitions,
                                                                      birth_death=self.bd_list)
            model = copy.deepcopy(self.piecewise_param_est_model)


        return model