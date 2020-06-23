import numpy as np


class ValueFunctionClassTabular:
    def __init__(self, num_states, num_actions, default_value=0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.default_value = default_value
        self.value_function_matrix = None

    def fit(self, state_action_array, target_value):
        value_function_matrix = (
            np.zeros([self.num_states, self.num_actions]))
        num_state_action_observation = (
            np.zeros([self.num_states, self.num_actions]))
        num_obs = state_action_array.shape[0]
        for obs_idx in range(num_obs):
            state = state_action_array[obs_idx, 0]
            action = state_action_array[obs_idx, 1]
            value_function_matrix[state, action] += target_value[obs_idx]
            num_state_action_observation[state, action] += 1
        value_function_matrix[np.where(num_state_action_observation == 0)] = (
            self.default_value)
        # This step is to ensure no division by 0
        num_state_action_observation[
            np.where(num_state_action_observation == 0)] = 1
        value_function_matrix /= num_state_action_observation
        self.value_function_matrix = value_function_matrix

    def predict(self, state_action_array):
        assert self.value_function_matrix is not None, (
            "ValueFunctionClassTabular.fit() must be called before calling ",
            "ValueFunctionClassTabular.predict()")
        num_obs = state_action_array.shape[0]
        predicted_value_arr = np.zeros(num_obs)
        for obs_idx in range(num_obs):
            state = state_action_array[obs_idx, 0]
            action = state_action_array[obs_idx, 1]
            predicted_value_arr[obs_idx] = (
                self.value_function_matrix[state, action])
        return predicted_value_arr

    def return_value_function_matrix(self):
        return self.value_function_matrix


