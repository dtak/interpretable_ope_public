import numpy as np
import matplotlib.pyplot as plt

def compute_influence_tensor_from_mdp_model(
        transition_tensor,
        policy_matrix,
        gamma,
        max_steps,
        plot_state_evolution_matrix=False,
        save_state_evolution_matrix_figure=False,
        influence_structure="s2s"):

    num_states, num_actions, _ = transition_tensor.shape
    # Propagate the state distribution given both policy and MDP transitions
    state_evolution_matrix = np.einsum(
        "ia, iaj -> ij", policy_matrix, transition_tensor)
    if plot_state_evolution_matrix:
        plt.figure()
        plt.imshow(state_evolution_matrix)
        plt.title("One step state evolution matrix")
        plt.xlabel("\"To\" state")
        plt.ylabel("\"From\" state")
        if save_state_evolution_matrix_figure:
            plt.savefig("tmp_state_evolution_matrix.eps",
                        bbox_inches="tight",
                        transparent=True)

    multi_step_state_transition_tensors = []
    if influence_structure == "s2s":
        multi_step_state_transition_tensors.append(np.eye(num_states))
        multi_step_state_transition_tensors.append(
            gamma * state_evolution_matrix)
    elif influence_structure == "s2sa":
        multi_step_state_transition_tensors.append(np.zeros(
            [num_states, num_states, num_actions]))
        for action_idx in range(num_actions):
            multi_step_state_transition_tensors[0][:, :, action_idx] = (
                np.eye(num_states))

        multi_step_state_transition_tensors.append(np.zeros(
            [num_states, num_states, num_actions]))
        for action_idx in range(num_actions):
            multi_step_state_transition_tensors[1][:, :, action_idx] = (
                transition_tensor[:, action_idx, :])
    elif influence_structure == "sa2sa":
        pass
    converged = False
    time_idx = 1
    while not converged:
        if influence_structure == "s2s":
            multi_step_state_transition_tensors.append(np.einsum(
                "kj, ik -> ij",
                gamma * state_evolution_matrix,
                multi_step_state_transition_tensors[-1]))
        elif influence_structure == "s2sa":
            multi_step_state_transition_tensors.append(np.zeros(
                [num_states, num_states, num_actions]))
            for action_idx in range(num_actions):
                multi_step_state_transition_tensors[-1][:, :, action_idx] = (
                    np.einsum(
                        "kj, ik -> ij",
                        gamma * state_evolution_matrix,
                        multi_step_state_transition_tensors[-2][
                            :, :, action_idx]))
        elif influence_structure == "sa2sa":
            pass
        time_idx += 1
        converged = time_idx >= max_steps - 1
    if influence_structure == "s2s":
        influence_tensor = np.zeros([num_states, num_states])
    elif influence_structure == "s2sa":
        influence_tensor = np.zeros([num_states, num_states, num_actions])
    for time_idx in range(len(multi_step_state_transition_tensors)):
        influence_tensor += multi_step_state_transition_tensors[time_idx]
    return influence_tensor
