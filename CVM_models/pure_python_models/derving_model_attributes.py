"""
Creation:
    Author: Martin Grunnill
    Date: 22/08/2022
Description: Derving pure python model attributes using pygom.
    
"""
import json

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
        state_index = {str(state): index
                       for index, state in enumerate(self.pygom_model.state_list)}
        states_by_length_descending = sorted(state_index.keys(),
                                             key=len, reverse=True)
        replacement = {state: 'y[' + str(state_index[state]) + ']'
                       for state in states_by_length_descending}
        params_by_length_descending = sorted(self.pygom_model._paramDict.keys(),
                                             key=len, reverse=True)
        replacement.update({parameter: "parameters['" + parameter + "']"
                            for parameter in params_by_length_descending})

        new_dict_of_keys = {}
        for coords, value in dict_of_keys.items():
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
    
        
        