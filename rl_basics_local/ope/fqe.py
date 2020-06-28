import rl_basics_local

import numpy as np

def fqe(transitions_data, policy, gamma, q_function, initial_states,
        print_progress=False, num_of_iterations = 100,
        return_q_function=False):
    '''
    :param transitions_data: rl_basics_local.classes.TransitionDataSet
    :param policy: function taking a state (as a numpy.ndarray) and return an
        action as a scalar
    :param gamma: discount factor for the MDP
    :param q_function: function class with .fit() and .predict() methods
    :param initial_states: list of states (as a numpy.ndarray)
    :return: policy value: estimated value of the policy
    '''
    # TODO : change to convergence check

    # Check that policy() returns the correct data type
    assert isinstance(transitions_data, rl_basics_local.classes.TransitionsDataSet), (
        "Error in fqe(): transitions_data must be a TransitionsDataSet")
    test_trans = transitions_data.transitions[0]
    test_action = policy(test_trans.state, time_step=0)
    assert isinstance(test_action, np.ndarray), (
        "policy() must return a numpy.ndarray")

    # Check that q_function.fit() works and q_function.predict() returns the
    # correct data type
    test_state_action_arr = []
    for trans in transitions_data.transitions[:2]:
        one_state_action_pair_arr = np.hstack([
            trans.state.reshape(1, -1),
            trans.action.reshape(1, -1)
        ])
        test_state_action_arr.append(one_state_action_pair_arr)
    test_state_action_arr = np.vstack(test_state_action_arr)
    test_target = np.zeros(test_state_action_arr.shape[0])
    q_function.fit(test_state_action_arr, test_target)
    test_q_function_prediction = q_function.predict(test_state_action_arr)
    assert isinstance(test_q_function_prediction, np.ndarray), (
        "q_function.predict() must return a numpy.ndarray")
    assert len(test_q_function_prediction.shape) == 1, (
        "q_function.predict() must return a 1D array")

    state_with_observed_action = []
    for trans in transitions_data.transitions:
        one_state_action_pair_arr = np.hstack([
            trans.state.reshape(1, -1),
            trans.action.reshape(1, -1)
        ])
        state_with_observed_action.append(one_state_action_pair_arr)
    state_with_observed_action = np.vstack(
        state_with_observed_action)

    next_state_with_policy_action = []
    for trans in transitions_data.transitions:
        one_state_action_pair_arr = np.hstack([
            trans.next_state.reshape(1, -1),
            policy(trans.next_state, time_step=trans.time_step).reshape(1, -1)
        ])
        next_state_with_policy_action.append(one_state_action_pair_arr)
    next_state_with_policy_action = np.vstack(
        next_state_with_policy_action)

    initial_states_with_policy_action = []
    for state in initial_states:
        one_state_action_pair_arr = np.hstack([
            state.reshape(1, -1),
            policy(state, time_step=0).reshape(1, -1)
        ])
        initial_states_with_policy_action.append(one_state_action_pair_arr)
    initial_states_with_policy_action = np.vstack(
        initial_states_with_policy_action)

    rewards = []
    for trans in transitions_data.transitions:
        rewards.append(trans.reward)
    rewards = np.vstack(rewards)[:, 0]

    target = np.zeros(state_with_observed_action.shape[0])
    q_function.fit(state_with_observed_action, target)

    # est_policy_value_list = []
    for idx_iter in range(num_of_iterations):
        if print_progress:
            print("Iteration {}".format(idx_iter))
        target = (
            rewards
            + gamma
            * q_function.predict(next_state_with_policy_action))
        q_function.fit(state_with_observed_action, target)
        # values_of_initial_states = q_function.predict(
        #     initial_states_with_policy_action)
        # est_policy_value_list.append(np.mean(values_of_initial_states))

    values_of_initial_states = q_function.predict(
        initial_states_with_policy_action)
    est_policy_value = np.mean(values_of_initial_states)

    # plt.plot(est_policy_value_list)
    # plt.show()

    if return_q_function:
        return est_policy_value, q_function
    else:
        return est_policy_value