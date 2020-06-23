import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

import rl_basics_local
from rl_basics_local.ope.ope import *

def influence_analysis__is(traj_dataset, policy_eval, policy_behavior, gamma):
    value_estimate, additional_output = ope_is(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        return_additional_output=True)
    num_traj = additional_output['num_traj']
    traj_weight_arr = additional_output['traj_weight_arr']
    traj_return_arr = additional_output['traj_return_arr']
    traj_influence_arr = np.zeros(num_traj)
    for idx_traj in range(num_traj):
        traj_influence_arr[idx_traj] = (
            (value_estimate
             - traj_weight_arr[idx_traj] * traj_return_arr[idx_traj])
            / (num_traj - 1))
    return value_estimate, traj_influence_arr

def influence_analysis__wis(traj_dataset, policy_eval, policy_behavior, gamma):
    value_estimate, additional_output = ope_wis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        return_additional_output=True)
    num_traj = additional_output['num_traj']
    traj_weight_arr = additional_output['traj_weight_arr']
    traj_return_arr = additional_output['traj_return_arr']
    sum_of_weights = additional_output['sum_of_weights']
    traj_influence_arr = np.zeros(num_traj)
    for idx_traj in range(num_traj):
        traj_influence_arr[idx_traj] = (
            (value_estimate - traj_return_arr[idx_traj])
            * (traj_weight_arr[idx_traj]
               / (sum_of_weights - traj_weight_arr[idx_traj])))
    return value_estimate, traj_influence_arr

def influence_analysis__pdis(traj_dataset, policy_eval, policy_behavior, gamma):
    value_estimate, additional_output = ope_pdis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        return_additional_output=True)
    num_traj = additional_output['num_traj']
    traj_contribution_to_estimator = (
        additional_output['traj_contribution_to_estimator'])
    traj_influence_arr = np.zeros(num_traj)
    for idx_traj in range(num_traj):
        traj_influence_arr[idx_traj] = (
            (value_estimate - traj_contribution_to_estimator[idx_traj])
            / (num_traj - 1))
    return value_estimate, traj_influence_arr

def test_influence_analysis(
        traj_dataset,
        policy_eval,
        policy_behavior,
        gamma,
        ope_func,
        influence_func):

    value_estimate, traj_influence_arr = influence_func(
        traj_dataset, policy_eval, policy_behavior, gamma)
    num_traj = traj_dataset.return_num_traj()
    true_influence_arr = np.zeros(num_traj)
    influence_error_arr = np.zeros(num_traj)
    for idx_traj in range(num_traj):
        if idx_traj % 10 == 0:
            print('Traj #{}'.format(idx_traj))
        traj_dataset_copy = deepcopy(traj_dataset)
        traj_dataset_copy.remove_trajectory(idx_traj)
        new_value_estimate = ope_func(
            traj_dataset_copy,
            policy_eval,
            policy_behavior,
            gamma)
        true_influence_arr[idx_traj] = new_value_estimate - value_estimate
        influence_error_arr[idx_traj] = (
            true_influence_arr[idx_traj] - traj_influence_arr[idx_traj])