import numpy as np


def compute_reward_ste_vector(transitions_data_, num_states, num_actions):
    rewards_observed_per_state_action = (
        [[] for state_idx in range(num_states)])
    for trans in transitions_data_.transitions:
        rewards_observed_per_state_action[trans.state[0]].append(trans.reward)
    reward_ste_vector = np.zeros(num_states)
    for state_idx in range(num_states):
        if len(rewards_observed_per_state_action[state_idx]) > 0:
            reward_ste_vector[state_idx] = (
                np.std(rewards_observed_per_state_action[state_idx])
                / len(rewards_observed_per_state_action[state_idx])**0.5)
        else:
            reward_ste_vector[state_idx] = np.nan
    reward_ste_vector[np.where(np.isnan(reward_ste_vector))] = (
        np.nanmax(reward_ste_vector))
    return reward_ste_vector


def compute_value_function_corruption_using_influence(
        influence_tensor, corruption_vector_, influence_structure="s2s"):
    if influence_structure == "s2s":
        value_function_corruption = np.dot(influence_tensor, corruption_vector_)
    elif influence_structure == "s2sa":
        # TODO: Can I turn this into an einsum?
        num_states, _, num_actions = influence_tensor.shape
        value_function_corruption = np.zeros(
            [num_states, num_actions])
        for action_idx in range(num_actions):
            value_function_corruption[:, action_idx] = (
                np.dot(influence_tensor[:, :, action_idx], corruption_vector_))
    return value_function_corruption


class SparseMatrix:
    def __init__(self, dim, M=None):
        self.non_zero_cols = {}
        self.non_zero_rows = {}
        self.dim = dim
        if M is not None:
            for row in range(M.shape[0]):
                for col in range(M.shape[1]):
                    if M[row, col] != 0:
                        self.assign_element(row, col, M[row, col])

    def __mul__(self, scalar):
        new_matrix = SparseMatrix(self.dim)
        for col in self.non_zero_cols.keys():
            for row in self.non_zero_cols[col].keys():
                value = self.non_zero_cols[col][row]
                new_matrix.assign_element(row, col, scalar * value)
        return new_matrix

    __rmul__ = __mul__

    def __add__(self, other_mat):
        assert isinstance(other_mat, SparseMatrix), (
            '+ operator only works on two SparseMatrix objects')
        new_matrix = SparseMatrix(self.dim)
        for col in self.non_zero_cols.keys():
            for row in self.non_zero_cols[col].keys():
                value = self.non_zero_cols[col][row]
                new_matrix.assign_element(row, col, value)

        for col in other_mat.non_zero_cols.keys():
            for row in other_mat.non_zero_cols[col].keys():
                value_to_add = other_mat.non_zero_cols[col][row]
                is_non_zero_element = (col in new_matrix.non_zero_cols.keys()
                    and row in new_matrix.non_zero_cols[col].keys())
                if is_non_zero_element:
                    old_value = new_matrix.non_zero_cols[col][row]
                else:
                    old_value = 0
                new_matrix.assign_element(row, col, old_value + value_to_add)
        return new_matrix

    def assign_element(self, row, col, val):
        if val == 0:
            print(0)
        if col not in self.non_zero_cols.keys():
            self.non_zero_cols[col] = {}
        if row not in self.non_zero_rows.keys():
            self.non_zero_rows[row] = {}
        self.non_zero_cols[col][row] = val
        self.non_zero_rows[row][col] = val

    def dot_right(self, other_mat):
        if isinstance(other_mat, np.ndarray):
            # other_mat is a dense matrix, returns dot(other_mat, S)
            assert other_mat.shape[1] == self.dim, 'dimension mismatch'
            product_mat = SparseMatrix(self.dim)
            for col in self.non_zero_cols.keys():
                for row in range(self.dim):
                    val = 0
                    for sum_idx in self.non_zero_cols[col].keys():
                        val += (
                            other_mat[row][sum_idx]
                            * self.non_zero_cols[col][sum_idx])
                    if val != 0:
                        product_mat.assign_element(row, col, val)
        elif isinstance(other_mat, SparseMatrix):
            # other_mat is a SparseMatrix, returns dot(other_mat, S)
            assert other_mat.dim == self.dim, 'dimension mismatch'
            product_mat = SparseMatrix(self.dim)
            for row in other_mat.non_zero_rows.keys():
                for col in self.non_zero_cols.keys():
                    temp_sum = 0
                    for dummy_idx in other_mat.non_zero_rows[row].keys():
                        if dummy_idx in self.non_zero_cols[col].keys():
                            temp_sum += (
                                other_mat.non_zero_rows[row][dummy_idx]
                                * self.non_zero_cols[col][dummy_idx]
                            )
                    if temp_sum != 0:
                        product_mat.assign_element(row, col, temp_sum)
        return product_mat

    def to_array(self):
        dense = np.zeros((self.dim, self.dim))
        for col in self.non_zero_cols.keys():
            for row in self.non_zero_cols[col].keys():
                dense[row, col] = self.non_zero_cols[col][row]
        return dense

    def remove_idx(self, idx_to_remove):
        new_matrix = SparseMatrix(self.dim - 1)
        for col in self.non_zero_cols.keys():
            for row in self.non_zero_cols[col].keys():
                value = self.non_zero_cols[col][row]
                if col < idx_to_remove and row < idx_to_remove:
                    new_matrix.assign_element(row, col, value)
                elif col > idx_to_remove and row < idx_to_remove:
                    new_matrix.assign_element(row, col - 1, value)
                elif col < idx_to_remove and row > idx_to_remove:
                    new_matrix.assign_element(row - 1, col, value)
                elif col > idx_to_remove and row > idx_to_remove:
                    new_matrix.assign_element(row - 1, col - 1, value)
        return new_matrix

    def add_to_dense(self, dense):
        for col in self.non_zero_cols.keys():
            for row in self.non_zero_cols[col].keys():
                dense[row, col] += self.non_zero_cols[col][row]
        return dense

