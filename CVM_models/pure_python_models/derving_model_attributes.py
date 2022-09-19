"""
Creation:
    Author: Martin Grunnill
    Date: 22/08/2022
Description: Derving pure python model attributes using pygom.
    
"""
import json
from collections import OrderedDict

class AttributeGetter:
    def __init__(self, pygom_model_constucter, metapopulation_structure, name_for_model, save_directory,
                 include_observed_states=True):
        self.pygom_model_constructor = pygom_model_constucter(metapopulation_structure,
                                                              include_observed_states=include_observed_states)
        self.pygom_model = self.pygom_model_constructor.generate_model()
        self.name_for_model = name_for_model
        self.save_directory = save_directory

    def _gen_mod_dok_matrix(self, sym_matrix):
        rows, cols = sym_matrix.shape
        dict_of_keys = {(row, col): str(sym_matrix[row, col])
                        for row in range(rows)
                        for col in range(cols)
                        if sym_matrix[row, col] != 0}
        state_index = {state + '_' + cluster + '_' + vaccine_group: 'y[' + str(index) + ']'
                       for cluster, vaccine_group_state_index in
                       self.pygom_model_constructor.cluste_vaccine_group_state_index.items()
                       for vaccine_group, state_index in vaccine_group_state_index.items()
                       for state, index in state_index.items()}
        states_by_length_descending = sorted(state_index.keys(),
                                             key=len, reverse=True)
        replacement = {state: 'y[' + str(state_index[state]) + ']'
                       for state in states_by_length_descending}
        param_index = {str(param): index
                       for index, param in enumerate(self.pygom_model.param_list)}
        params_by_length_descending = sorted(param_index.keys(),
                                             key=len, reverse=True)

        replacement.update({param: "parameters['" + str(param_index[param]) + "']"
                            for param in params_by_length_descending})
        temp_dict_of_keys = {}
        for coords, value in dict_of_keys.items():
            for orignal, new in replacement.items():
                value = value.replace(orignal, new)
            temp_dict_of_keys[coords] = value

        index_param = {value: key for key, value in param_index.items()}
        index_param = OrderedDict(sorted(index_param.items(), reverse=True))
        replacement = {"parameters['" + str(index) + "']": "parameters['" + param + "']"
                       for index, param in index_param.items()}
        new_dict_of_keys = {}
        for coords, value in temp_dict_of_keys.items():
            for orignal, new in replacement.items():
                value = value.replace(orignal, new)
            new_dict_of_keys[coords] = value



        return {str(key): value for key, value in
                new_dict_of_keys.items()}  # json keys cannot be tuples for this so convert to string

    def save_odes(self):
        odes = self.pygom_model.get_ode_eqn()
        odes_dok = self._gen_mod_dok_matrix(odes)
        with open(self.save_directory + self.name_for_model +"_odes.json", "w") as ode_outfile:
            json.dump(odes_dok, ode_outfile)

    def save_jacobian(self):
        jacobian = self.pygom_model.get_jacobian_eqn()
        jacobian_dok = self._gen_mod_dok_matrix(jacobian)
        with open(self.save_directory + self.name_for_model +"_jacobian.json", "w") as jac_outfile:
            json.dump(jacobian_dok, jac_outfile)

    def save_gradient(self):
        gradient = self.pygom_model.get_grad_eqn()
        gradient_dok = self._gen_mod_dok_matrix(gradient)
        with open(self.save_directory + self.name_for_model +"_gradient.json", "w") as grad_outfile:
            json.dump(gradient_dok, grad_outfile)

    def save_diff_jacobian(self):
        diff_jacobian = self.pygom_model.get_diff_jacobian_eqn()
        diff_jacobian_dok = self._gen_mod_dok_matrix(diff_jacobian)
        with open(self.save_directory + self.name_for_model +"_diff_jacobian.json", "w") as diff_jac_outfile:
            json.dump(diff_jacobian_dok, diff_jac_outfile)

    
    def save_all_attributes(self):
        self.save_odes()
        self.save_jacobian()
        self.save_gradient()
        self.save_diff_jacobian()
    
        
        