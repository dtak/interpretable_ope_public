import numpy as np
import matplotlib.pyplot as plt

from rl_basics_local import generate_trajectory


def trans_weight(trans, policy_eval, policy_behavior):
    if policy_eval.return_proba(
            trans.state, trans.action, trans.time_step) == 0:
        trans_is_weight = 0
    else:
        if policy_behavior.return_proba(
                trans.state, trans.action, trans.time_step) == 0:
            raise Exception('pi_b = 0 but pi_e != 0')
        trans_is_weight = (
            policy_eval.return_proba(trans.state, trans.action, trans.time_step)
            / policy_behavior.return_proba(
                trans.state, trans.action, trans.time_step))
    return trans_is_weight

def traj_weight(traj, policy_eval, policy_behavior):
    weight = 1
    for trans in traj.transitions:
        weight *= trans_weight(trans, policy_eval, policy_behavior)
    return weight


def ope_is(traj_dataset, policy_eval, policy_behavior, gamma,
           return_additional_output=False):
    num_traj = traj_dataset.return_num_traj()
    traj_weight_arr = np.zeros(num_traj)
    traj_return_arr = np.zeros(num_traj)
    for idx_traj, traj in enumerate(traj_dataset.trajectories):
        traj_weight_arr[idx_traj] = traj_weight(
            traj, policy_eval, policy_behavior)
        traj_return_arr[idx_traj] = traj.return_trajectory_return(gamma)
    value_estimate = np.sum(traj_weight_arr * traj_return_arr) / num_traj
    if not return_additional_output:
        return value_estimate
    else:
        additional_output = {
            'num_traj': num_traj,
            'traj_weight_arr': traj_weight_arr,
            'traj_return_arr': traj_return_arr
        }
        return value_estimate, additional_output

def ope_wis(traj_dataset, policy_eval, policy_behavior, gamma,
            return_additional_output=False):
    num_traj = traj_dataset.return_num_traj()
    traj_weight_arr = np.zeros(num_traj)
    traj_return_arr = np.zeros(num_traj)
    for idx_traj, traj in enumerate(traj_dataset.trajectories):
        traj_weight_arr[idx_traj] = traj_weight(
            traj, policy_eval, policy_behavior)
        traj_return_arr[idx_traj] = traj.return_trajectory_return(gamma)
    sum_of_weights = np.sum(traj_weight_arr)
    if sum_of_weights == 0:
        value_estimate = 0
    else:
        value_estimate = (
                np.sum(traj_weight_arr * traj_return_arr) / sum_of_weights)
    if not return_additional_output:
        return value_estimate
    else:
        additional_output = {
            'num_traj': num_traj,
            'traj_weight_arr': traj_weight_arr,
            'traj_return_arr': traj_return_arr,
            'sum_of_weights': sum_of_weights
        }
        return value_estimate, additional_output

def ope_pdis(traj_dataset,
             policy_eval,
             policy_behavior,
             gamma,
             return_additional_output=False):
    num_traj = traj_dataset.return_num_traj()
    traj_contribution_to_estimator = np.zeros(num_traj)
    for idx_traj, traj in enumerate(traj_dataset.trajectories):
        current_contribuiton = 0
        is_weight = 1
        discount = 1
        for trans in traj.transitions:
            is_weight *= trans_weight(trans, policy_eval, policy_behavior)
            current_contribuiton += discount * is_weight * trans.reward
            discount *= gamma
        traj_contribution_to_estimator[idx_traj] = current_contribuiton
    value_estimate = np.sum(traj_contribution_to_estimator) / num_traj
    if not return_additional_output:
        return value_estimate
    else:
        additional_output = {
            'num_traj': num_traj,
            'traj_contribution_to_estimator': traj_contribution_to_estimator
        }
        return value_estimate, additional_output

def ope_dr(traj_dataset, policy_eval, policy_behavior, gamma,
           v_est, q_est,
           return_additional_output=False):
    ''' Haven't checked that this function works!!!!!!!! '''
    num_traj = traj_dataset.return_num_traj()
    traj_contribution_to_estimator = np.zeros(num_traj)


    print()
    print()
    print()
    print()
    total_est = 0
    total_est_list =[]


    for idx_traj, traj in enumerate(traj_dataset.trajectories):

        print()
        print(total_est)
        print('-------------')
        total_est = 0

        current_contribuiton = 0
        is_weight = 1
        discount = 1
        for trans in traj.transitions:
            state = int(trans.state[0])
            action = int(trans.action[0])
            old_is_weight = is_weight
            is_weight *= trans_weight(trans, policy_eval, policy_behavior)
            current_pdis_contribuiton = is_weight * trans.reward
            current_est_contribuiton = (
                    is_weight * q_est[state, action]
                    - old_is_weight * v_est[state])

            print(current_est_contribuiton)
            total_est += current_est_contribuiton

            current_contribuiton += discount * (
                current_pdis_contribuiton
                - current_est_contribuiton)
            discount *= gamma
        traj_contribution_to_estimator[idx_traj] = current_contribuiton

        total_est_list.append(total_est)
    plt.figure()
    plt.hist(total_est_list, 30)
    plt.title('normal')


    value_estimate = np.sum(traj_contribution_to_estimator) / num_traj
    if not return_additional_output:
        return value_estimate
    else:
        additional_output = {
            'num_traj': num_traj,
            'traj_contribution_to_estimator': traj_contribution_to_estimator
        }
        return value_estimate, additional_output

def ope(traj_dataset, policy_eval, policy_behavior, gamma, method,
        v_est=None, q_est=None):
    '''' General function to call the different OPE methods '''
    if method == 'is':
        return ope_is(traj_dataset, policy_eval, policy_behavior, gamma)
    elif method == 'wis':
        return ope_wis(traj_dataset, policy_eval, policy_behavior, gamma)
    elif method == 'pdis':
        return ope_pdis(traj_dataset, policy_eval, policy_behavior, gamma)
    elif method == 'dr':
        assert v_est is not None and q_est is not None, (
            'Estimates of V and Q must be passed for DR estimation.')
        return ope_dr(traj_dataset, policy_eval, policy_behavior, gamma,
           v_est, q_est)




def parametric_model(transitions_data, policy, gamma,
                     transition_function,
                     reward_function,
                     initial_states,
                     is_done,
                     num_trajectories_to_generate=1):

    transition_function.train(transitions_data)
    reward_function.train(transitions_data)
    model_env = EnvParametricModel(
        transition_function, reward_function, initial_states, is_done)
    trajectories_return_list = []
    for traj_idx in range(num_trajectories_to_generate):
        trajectory = generate_trajectory(model_env, policy)
        trajectories_return_list.append(
            trajectory.return_trajectory_return(gamma))
    est_policy_value = np.mean(trajectories_return_list)

    return est_policy_value