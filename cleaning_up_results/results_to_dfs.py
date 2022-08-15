"""
Creation:
    Author: Martin Grunnill
    Date: 15/08/2022
Description: Module with functions for putting results into dataframes.
    
"""
import pandas as pd


def results_array_to_df(results, state_index, index=None):
    """
    Converts array of results from CVM models into a dataframe with multi-index columns.
    """
    multi_columns = []
    for cluster, sub_dict in state_index.items():
        if cluster != 'observed_states':
            for vaccine_group, state_dict in sub_dict.items():
                for state in state_dict.keys():
                    multi_columns.append((cluster, vaccine_group, state))
        else:
            for state in sub_dict.keys():
                multi_columns.append((cluster, None, state))
    
    results_df = pd.DataFrame(results, index=index)
    results_df.columns = pd.MultiIndex.from_tuples(multi_columns)
    return results_df

def results_df_pivoted(results_df):
    """
    Pivots CVM model results dataframe for use with seaborne.
    """
    results_melted = pd.melt(results_df, ignore_index=False)
    results_line_list = results_melted.reset_index()
    results_line_list.columns = ['time','cluster', 'vaccine_group','state','population']
    results_line_list.replace({'observed_states':'acumelated_totals'}, inplace=True)
    return results_line_list