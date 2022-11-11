"""
Creation:
    Author: Martin Grunnill
    Date: 28/08/2022
Description: Derving R0 for mass gathering model assuming one cluster and no vaccinations.
    
"""
from CVM_models.pygom_models.mass_gathering_vaccination import MGModelConstructor
from pygom import Transition, DeterministicOde, R0
import pygom

group_structure = {'clusters': '_',
                   'vaccine groups': '_'}

mg_model_constuctor = MGModelConstructor(group_structure)
mg_model = mg_model_constuctor.generate_model(variety='deterministic')
odes = mg_model.get_ode_eqn()
graph_dot = mg_model.get_transition_graph(show=False)
graph_dot.render(filename='single_pop_no_vaccine', format='pdf')
mg_model.state_list
disease_states = ['E____',
                  'G_I____','G_A____',
                  'P_I____','P_A____',
                  'M_H____','M_I____','M_A____',
                  'F_H____','F_I____','F_A____'
                  ]
#model_R0 = R0(mg_model, disease_states) #  silenced as it causes NameError: name 'lambda__' is not defined
#%%

# Will have to redefince model. May as well remove the suffixes '__' and '____' as well.
all_params = [item.removesuffix("____")
              for item in mg_model_constuctor.all_parameters]
all_params = [item.removesuffix("__")
              for item in all_params]
# parameters governing vaccination efficacy are not neeeded
all_params = [item
              for item in all_params
              if item not in ['l','h', 's']]

all_states = [item.removesuffix("____")
              for item in mg_model_constuctor.all_states]
new_transitions = []
derived_parameter = mg_model_constuctor.derived_params[0]
for transition in mg_model_constuctor.transitions:
    origin = transition.origin.removesuffix("____")
    destination = transition.destination.removesuffix("____")
    equation = transition.equation.replace(derived_parameter[0],derived_parameter[1])
    equation = equation.replace("__", '')
    # parameters governing vaccination efficacy are not neeeded
    equation = equation.replace("(1-l)*", '')
    equation = equation.replace("(1-h)*", '')
    equation = equation.replace("(1-s)*", '')
    equation = equation.replace("* (1 -l)", '')
    equation = equation.replace("* (1 -h)", '')
    equation = equation.replace("* (1 -s)", '')
    new_transitions.append(Transition(origin=origin,destination=destination,
                                      equation=equation,transition_type='T'))

mg_model_redefined = DeterministicOde(state=all_states,
                                      param=all_params,
                                      transition=new_transitions)
graph_dot = mg_model_redefined.get_transition_graph(show=False)
graph_dot.render(filename='single_pop_no_vaccine', format='pdf')
#%%
disease_states = [disease_state.removesuffix('____') for disease_state in disease_states]
model_R0 = R0(ode=mg_model_redefined, disease_state=disease_states)
# This suggests the R0 is 0. Which we now is wrong.
F_matrix , V_matrix = pygom.model.epi_analysis.disease_progression_matrices(ode=mg_model_redefined, disease_state=disease_states)
pygom_model_R0 = pygom.model.epi_analysis.R0_from_matrix(F_matrix , V_matrix)


#%%
import sympy
for list_of_symbols in [all_params, all_states]:
    for symbol in list_of_symbols:
        exec(symbol + ' = sympy.symbols("'+symbol +'")')
odes = mg_model_redefined.get_ode_eqn()
conv_odes = []
for ode in odes:
    eval('conv_odes.append('+str(ode)+')')
odes = sympy.Matrix(conv_odes)
infecteds = odes[1:-1]
infecteds = sympy.Matrix(odes[1:-1])
infecteds = infecteds.subs(S, N)
infecteds_jacobian = infecteds.jacobian(X=[E,
                                           G_I, G_A,
                                           P_I, P_A,
                                           M_H, M_I, M_A,
                                           F_H, F_I, F_A
                                           ])


# e.g. removing people becoming infected from the jacobian above.
Sigma = infecteds_jacobian.subs(beta, 0)
Sigma

# Obtainning matrix  of transitions into of infectious stages (T)
# E.g. removing people transitioning from the jacobian above.
# Suggest not use T to name a variable could be confused with transpose of a matrix.
T_inf_births_subs = {eval(param):0
                     for param in all_params
                     if param not in ['beta', 'theta', 'kappa']}
T_inf_births = infecteds_jacobian.subs(T_inf_births_subs)
T_inf_births

# Obtainning Next Geneation Matrix
Sigma_inv = Sigma**-1 # note for powers in python it is ** not ^.
neg_Sigma_inv = -Sigma_inv

K_L = T_inf_births*neg_Sigma_inv
K_L

# Finally the Basic Reproductive Number
eigen_values = K_L.eigenvals()
eigen_values
none_zero_eigen_values = [item for item in eigen_values.keys() if item !=0]
sympy_R0 = none_zero_eigen_values[0]
#%%
sympy_R0 = sympy.simplify(sympy_R0)

#%%
# Dervining Beta

R0 = sympy.symbols('R0')
eq_R0 = sympy.Eq(sympy_R0, R0)
beta_eq = sympy.solve(eq_R0, beta)
beta_eq = beta_eq[0]
#%%
beta_eq = sympy.simplify(beta_eq)