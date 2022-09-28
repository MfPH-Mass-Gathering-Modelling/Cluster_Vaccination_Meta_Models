"""
Creation:
    Author: Martin Grunnill
    Date: 28/09/2022
Description:Functions for calculating Reproductive numbers or Beta for Cluster Vaccination models.
    
"""


def MGE_R0_no_vaccine_1_cluster(beta, kappa, theta,
                                epsilon_3, epsilon_H,
                                gamma_A_1, gamma_A_2, gamma_I_1, gamma_I_2,
                                p_d, p_h, p_s):
    """ Calculates R0 for Mass gathering event assuming 1 cluster
        (or homogenous mixing between clusters) and no vaccination.
        See Dervinng_R0_and_beta_for_1_cluster_no_vaccine.py for dervation.
    """
    R0 = (-beta * kappa * p_d * p_h * p_s / gamma_I_2 +
          beta * kappa * p_d * p_s / gamma_I_2 -
          beta * p_d * p_s / gamma_I_2 +
          beta * p_s / gamma_I_2 -
          beta * kappa * p_d * p_h * p_s / gamma_I_1 +
          beta * kappa * p_d * p_s / gamma_I_1 -
          beta * p_d * p_s / gamma_I_1 +
          beta * p_s / gamma_I_1 -
          beta * p_s * theta / gamma_A_2 +
          beta * theta / gamma_A_2 -
          beta * p_s * theta / gamma_A_1 +
          beta * theta / gamma_A_1 +
          beta * kappa * p_d * p_h * p_s / epsilon_H +
          beta * theta / epsilon_3)
    return R0

def MGE_beta_no_vaccine_1_cluster(R_0, kappa, theta,
                                  epsilon_3, epsilon_H,
                                  gamma_A_1, gamma_A_2, gamma_I_1, gamma_I_2,
                                  p_d, p_h, p_s):
    """ Calculates beta given R0 for Mass gathering event assuming 1 cluster
        (or homogenous mixing between clusters) and no vaccination.
        See Dervinng_R0_and_beta_for_1_cluster_no_vaccine.py for dervation.
    """

    numrator = R_0 * epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * gamma_I_2
    denominator = (-epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * kappa * p_d * p_h * p_s +
                   epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * kappa * p_d * p_s - epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * p_d * p_s +
                   epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * p_s - epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_2 * kappa * p_d * p_h * p_s +
                   epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_2 * kappa * p_d * p_s - epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_2 * p_d * p_s +
                   epsilon_3 * epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_2 * p_s - epsilon_3 * epsilon_H * gamma_A_1 * gamma_I_1 * gamma_I_2 * p_s * theta +
                   epsilon_3 * epsilon_H * gamma_A_1 * gamma_I_1 * gamma_I_2 * theta - epsilon_3 * epsilon_H * gamma_A_2 * gamma_I_1 * gamma_I_2 * p_s * theta +
                   epsilon_3 * epsilon_H * gamma_A_2 * gamma_I_1 * gamma_I_2 * theta + epsilon_3 * gamma_A_1 * gamma_A_2 * gamma_I_1 * gamma_I_2 * kappa * p_d * p_h * p_s +
                   epsilon_H * gamma_A_1 * gamma_A_2 * gamma_I_1 * gamma_I_2 * theta)
    return numrator/denominator