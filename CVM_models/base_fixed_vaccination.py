"""
Creation:
    Author: Martin Grunnill
    Date: 01/04/2022
Description:
    
"""
import copy
from pygom import TransitionType, Transition, DeterministicOde, SimulateOde


class BaseMultiClusterVacConstructor:
    beta_list = []
    lambda_dict = {}
    cluster_pops = {}
    transmission_combos = []
    all_states = []
    all_parameters = []
    transitions = []
    bd_list = []
    derived_params = []
    states = []
    vaccinable_states = []
    infectious_states = []
    symptomatic_states = []
    contactable_states = []
    non_specific_params = []
    vaccine_specific_params = []
    cluster_specific_params = []
    vaccine_and_cluster_specific_params = ['nu']
    stoc_model = None
    det_model = None

    def __init__(self, clusters, vaccine_groups):
        self.infectious_and_symptomatic_states = [state for state in self.infectious_states
                                                  if state in self.symptomatic_states]
        self.infectious_and_asymptomatic_states = [state for state in self.infectious_states
                                                   if state not in self.symptomatic_states]
        self.clusters = clusters
        self.vaccine_groups = vaccine_groups
        self.vaccine_specific_params_dict = {vaccine_specific_param: []
                                             for vaccine_specific_param in
                                             self.vaccine_specific_params}
        self.cluster_specific_params_dict = {cluster_specific_param: []
                                              for cluster_specific_param in
                                              self.cluster_specific_params}
        self.vaccine_and_cluster_specific_params_dict = {vaccine_and_cluster_specific_param: []
                                                          for vaccine_and_cluster_specific_param in
                                                          self.vaccine_and_cluster_specific_params}

        self.states_dict = {state: [] for state in self.states}
        # Setting up clusters
        self.cluster_contactable_population = {}
        self.cluster_dead_pops = {}
        self.cluster_living_and_dead_pops = {}
        for cluster_i in self.clusters:
            cluster_pop = []
            cluster_dead = []
            for state in self.generic_states:
                for vaccine_group in self.vaccine_groups:
                    state_in_cluster = state + "_" + cluster_i + "_" + vaccine_group
                    self.all_states.append(state_in_cluster)
                    self.states_dict[state].append(state_in_cluster)
                    if state in self.living_states:
                        cluster_pop.append(state_in_cluster)
                    else:
                        cluster_dead.append(state_in_cluster)
            self.cluster_contactable_population[cluster_i] = '+'.join(cluster_pop)
        # setting up forces of infection
        self.append_semetric_transmission_from_list(self.clusters)
        self.derived_params = [('lambda_' + cluster_i, '+'.join(self.lambda_dict[cluster_i]))
                               for cluster_i in self.clusters]
        self._append_vaccination_group_transitions()

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
            temp_lambda = ['theta*' + infectous_state + '_' + cluster_j + '_' + vaccine_group
                           for infectous_state in self.infectous_non_symptomatic_states for vaccine_group in
                           self.vaccine_groups]
            temp_lambda += [infectous_state + '_' + cluster_j + '_' + vaccine_group
                            for infectous_state in self.infectous_symptomatic_states for vaccine_group in
                            self.vaccine_groups]
            beta = 'beta_' + cluster_i + '_' + cluster_j
            self.beta_list.append(beta)
            if cluster_i not in self.lambda_dict:
                self.lambda_dict[cluster_i] = []
            self.lambda_dict[cluster_i].append(
                '(' + '+'.join(temp_lambda) + ')*' + beta + '/(' + self.cluster_contactable_population[cluster_j] + ')')

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
                    self.append_semetric_transmission(self, cluster_i, cluster_j)
                elif include_intra:
                    self.append_intra_transmission(cluster_i)

    def _append_vaccination_group_transitions(self):
        for vaccine_stage, vaccine_group in enumerate(self.vaccine_groups):
            for cluster in self.clusters:
                if vaccine_group is not self.vaccine_groups[-1]: # following transitions do not occur in last vaccine group.
                    vaccine_group_plus_1 = self.vaccine_groups[vaccine_stage+1]
                    nu_i_v = 'nu_' + cluster + '_' + vaccine_group
                    self.vaccine_and_cluster_specific_params_dict['nu'].append(nu_i_v)
                    for state in self.vaccinable_states:
                        state_i_v = state + "_" + cluster + '_' + vaccine_group
                        state_i_v_plus_1 = state + "_" + cluster + '_' + vaccine_group_plus_1
                        self.transitions.append(Transition(origin=state_i_v,
                                                           destination=state_i_v_plus_1,
                                                           equation= nu_i_v+ '*' + state_i_v,
                                                           transition_type=TransitionType.T)
                                                )

    def attach_all_params_list(self):
        """
        Create a list of all parameters and attach it.
        """
        self.all_parameters += self.non_specific_params + self.beta_list
        dictionary_list = [
            self.vaccine_specific_params_dict,
            self.cluster_specific_params_dict,
            self.vaccine_and_cluster_specific_params_dict
        ]
        for specific_params_dict in dictionary_list:
            for list_item in specific_params_dict.values():
                self.all_parameters += list_item

    def generate_model(self, stochastic=False):
        """Generate pygom based model.
        """
        if stochastic:
            if self.stoc_model is None:
                self.stoc_model = SimulateOde(self.all_states,
                                              self.all_parameters,
                                              derived_param=self.derived_params,
                                              transition=self.transitions,
                                              birth_death=self.bd_list)
            model = copy.deepcopy(self.stoc_model)
        else:
            if self.det_model is None:
                self.det_model = DeterministicOde(self.all_states,
                                                  self.all_parameters,
                                                  derived_param=self.derived_params,
                                                  transition=self.transitions,
                                                  birth_death=self.bd_list)
            model = copy.deepcopy(self.det_model)

        return model