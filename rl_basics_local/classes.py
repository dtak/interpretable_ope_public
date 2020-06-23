import numpy as np


class Transition:
    def __init__(self, state, action, reward, next_state,
                 time_step=None, latent_state=None):
        # Initialize state
        assert isinstance(state, np.ndarray), (
            "Transition __init__ error: state is not a numpy.ndarray")
        assert len(state.shape) == 1, (
            "Transition __init__ error: state is not a 1D array")
        self.state = state  # 1D numpy array

        # Initialize action
        assert isinstance(action, np.ndarray), (
            "Transition __init__ error: action is not a numpy.ndarray")
        assert len(action.shape) == 1, (
            "Transition __init__ error: action is not a 1D array")
        self.action = action  # 1D numpy array

        # Initialize reward
        assert np.isscalar(reward), (
            "Transition __init__ error: reward is not a numpy scalar")
        self.reward = reward + 0.0  # numpy float

        # Initialize next_state
        assert isinstance(next_state, np.ndarray), (
            "Transition __init__ error: next_state is not a numpy.ndarray")
        assert len(next_state.shape) == 1, (
            "Transition __init__ error: next_state is not a 1D array")
        self.next_state = next_state  # 1D numpy array

        # Initialize time
        if time_step is not None:
            assert isinstance(time_step, int), (
                "Transition __init__ error: time_step is not an integer")
        self.time_step = time_step

        # Initialize latent_state
        if latent_state is not None:
            assert isinstance(latent_state, np.ndarray), (
                "Transition __init__ error:",
                "latent_state is not a numpy.ndarray")
            assert len(latent_state.shape) == 1, (
                "Transition __init__ error: latent_state is not a 1D array")
        self.latent_state = latent_state

    def __repr__(self):
        np.set_printoptions(precision=2)
        repr_str = (
            "State: {} --- Action: {}"
            "--- Reward: {:.2f} --- Next state: {}").format(
                self.state,
                self.action,
                self.reward,
                self.next_state)
        if self.time_step is not None:
            repr_str = '{} - '.format(self.time_step) + repr_str
        if self.latent_state is not None:
            repr_str += ' --- Latent state: {}'.format(self.latent_state)
        return repr_str

    def set_latent_state(self, latent_state):
        assert isinstance(latent_state, np.ndarray), (
            "Transition set_latent_state() error:",
            "latent_state is not a numpy.ndarray")
        assert len(latent_state.shape) == 1, (
            "Transition set_latent_state() error:",
            "latent_state is not a 1D array")
        self.latent_state = latent_state


class TransitionsDataSet:
    def __init__(self):
        self.transitions = []

    def add_transition(self, transition):
        assert isinstance(transition, Transition), (
            "Error in TransitionsDataSet.add_transition(): "
            "Input must be a Transition")
        self.transitions.append(transition)

    def add_trajectory(self, trajectory):
        assert isinstance(trajectory, Trajectory), (
            "Error in TransitionsDataSet.add_trajectory(): "
            "Input must be a Trajectory")
        for trans in trajectory.transitions:
            self.add_transition(trans)

    def remove_transition(self, idx_of_transition_to_remove):
        del self.transitions[idx_of_transition_to_remove]

    def return_num_trans(self):
        return len(self.transitions)

    def move_transition_to_end_of_list(self, idx_of_transition_to_move):
        self.transitions.append(self.transitions.pop(idx_of_transition_to_move))

    def return_states_extremes(self):
        min_states_values = self.transitions[0].state
        max_states_values = self.transitions[0].state
        for trans in self.transitions:
            state = trans.state
            next_state = trans.next_state
            min_states_values = np.minimum(min_states_values, state)
            max_states_values = np.maximum(max_states_values, state)
            min_states_values = np.minimum(min_states_values, next_state)
            max_states_values = np.maximum(max_states_values, next_state)
        return min_states_values, max_states_values

    def return_state_dimensionality(self):
        return len(self.transitions[0].state)


class Trajectory:
    def __init__(self):
        self.transitions = []

    def add_transition(self, trans):
        self.transitions.append(trans)

    def __repr__(self):
        final_repr = ""
        # for trans_idx in range(len(self.transitions)):
        #     trans = self.transitions[trans_idx]
        #     final_repr += str(trans_idx) + " - " + repr(trans) + "\n"
        for trans in self.transitions:
            final_repr += repr(trans) + "\n"
        return final_repr

    def return_trajectory_return(self, gamma):
        traj_return = 0
        discout = 1
        for trans in self.transitions:
            traj_return += trans.reward * discout
            discout *= gamma
        self.trajectory_return = traj_return
        return traj_return

    def return_traj_len(self):
        return len(self.transitions)

    def return_initial_state(self):
        return self.transitions[0].state

    def return_states_extremes(self):
        min_states_values = self.transitions[0].state
        max_states_values = self.transitions[0].state
        for trans in self.transitions:
            state = trans.state
            next_state = trans.next_state
            min_states_values = np.minimum(min_states_values, state)
            max_states_values = np.maximum(max_states_values, state)
            min_states_values = np.minimum(min_states_values, next_state)
            max_states_values = np.maximum(max_states_values, next_state)
        return min_states_values, max_states_values


class TrajectoriesDataSet:
    def __init__(self):
        self.trajectories = []

    def add_trajectory(self, trajectory):
        assert isinstance(trajectory, Trajectory), (
            "Error in TrajectoriesDataSet.add_trajectory(): "
            "Input must be a Transition")
        self.trajectories.append(trajectory)

    def remove_trajectory(self, idx_of_trajectory_to_remove):
        del self.trajectories[idx_of_trajectory_to_remove]

    def return_num_traj(self):
        return len(self.trajectories)

    def return_as_TransitionsDataSet(self):
        trans_dataset = TransitionsDataSet()
        for traj in self.trajectories:
            trans_dataset.add_trajectory(traj)
        return trans_dataset


''' General policies '''

class PolicyTabularValueFunction:
    def __init__(self, tabular_value_function, eps_behavior=0):
        self.tabular_value_function = tabular_value_function
        self.eps_behavior = eps_behavior
        self.num_actions = self.tabular_value_function.shape[1]

    def __call__(self, state, time_step=None):
        if np.random.rand() < self.eps_behavior:
            return np.array([np.random.choice(self.num_actions)])
        else:
            return np.array(
                [np.argmax(self.tabular_value_function[state[0], :])])

    def return_policy_matrix(self):
        num_states, num_actions = self.tabular_value_function.shape
        policy_matrix = (
            np.ones([num_states, num_actions])
            * self.eps_behavior/ num_actions)
        for state_idx in range(num_states):
            policy_matrix[
                state_idx,
                np.argmax(self.tabular_value_function[state_idx, :])] += (
                    1 - self.eps_behavior)
        return policy_matrix

