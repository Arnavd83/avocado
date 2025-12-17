#!/usr/bin/env python3
import asyncio
import json
import time
import random
import itertools
import argparse
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append("../..")
from compute_utilities.compute_utilities import compute_utilities
from compute_utilities.utils import convert_numpy
from compute_utilities.templates import comparison_prompt_template_default, comparison_prompt_template_reasoning_default



# -------------------------
# Helper Functions
# -------------------------

def extract_partial_sequences(sequence):
    """
    Extract all contiguous partial sequences from a full rollout.

    Args:
        sequence (List[str]): Full rollout as a list of observations.

    Returns:
        List[Tuple[str]]: List of partial sequences as tuples.
    """
    partial_sequences = []
    for i in range(1, len(sequence)+1):
        partial = tuple(sequence[:i])
        partial_sequences.append(partial)
    return partial_sequences


# -------------------------
# MarkovProcess Classes
# -------------------------

class State:
    def __init__(self, state_id: int, observation: str, is_terminal: bool = False):
        self.id = state_id
        self.observation = observation
        self.is_terminal = is_terminal

    def __repr__(self):
        return f"State(id={self.id}, observation='{self.observation}', is_terminal={self.is_terminal})"


class MarkovProcess:
    def __init__(
        self,
        states: list,
        transitions: dict,
        start_state_distribution: list
    ):
        """
        Initializes the Markov Process.

        :param states: List of State objects.
        :param transitions: Dictionary mapping state_id to a list of tuples (next_state_id, probability).
        :param start_state_distribution: List of tuples (state_id, probability) representing the start state distribution.
        """
        self.states = {state.id: state for state in states}
        self.transitions = transitions
        self.start_state_distribution = start_state_distribution

    def sample_start_state(self) -> int:
        """
        Samples a start state based on the start state distribution.

        :return: state_id of the sampled start state.
        """
        state_ids, probabilities = zip(*self.start_state_distribution)
        sampled_state = random.choices(state_ids, weights=probabilities, k=1)[0]
        return sampled_state

    def sample_sequence(self, max_steps: int = None) -> list:
        """
        Samples an observation sequence from the Markov Process.

        :param max_steps: Maximum number of steps to sample. If None, continue until a terminal state is reached.
        :return: List of observations.
        """
        current_state_id = self.sample_start_state()
        sequence = [self.states[current_state_id].observation]
        steps = 0

        while not self.states[current_state_id].is_terminal:
            if current_state_id not in self.transitions:
                break  # No transitions defined; treat as terminal

            next_states = self.transitions[current_state_id]
            if not next_states:
                break  # No possible transitions; treat as terminal

            next_state_ids, probabilities = zip(*next_states)
            next_state_id = random.choices(next_state_ids, weights=probabilities, k=1)[0]
            sequence.append(self.states[next_state_id].observation)
            current_state_id = next_state_id
            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

        return sequence

    def __repr__(self):
        return f"MarkovProcess(num_states={len(self.states)}, start_state_distribution={self.start_state_distribution})"


# -------------------------
# Markov Process Loading
# -------------------------

def define_markov_processes(env_file_path):
    """
    Loads and returns Markov Processes from a JSON file.

    Args:
        env_file_path (str): Path to the JSON file containing Markov Processes definitions.

    Returns:
        List[MarkovProcess]: List of MarkovProcess instances.
    """
    if not os.path.exists(env_file_path):
        raise FileNotFoundError(f"Environment file not found at {env_file_path}")
    
    with open(env_file_path, 'r') as f:
        data = json.load(f)

    markov_processes = []
    
    for mp_data in data.get("MarkovProcesses", []):
        name = mp_data.get("name", "Unnamed Process")
        states = []
        for state in mp_data.get("states", []):
            states.append(State(
                state_id=state["id"],
                observation=state["observation"],
                is_terminal=state.get("is_terminal", False)
            ))
        
        transitions = {}
        for state_id, trans_list in mp_data.get("transitions", {}).items():
            transitions[int(state_id)] = [[int(next_id), prob] for next_id, prob in trans_list]
        
        start_state_distribution = [[int(state_id), prob] for state_id, prob in mp_data.get("start_state_distribution", [])]
        
        mp = MarkovProcess(states, transitions, start_state_distribution)
        markov_processes.append(mp)

    return markov_processes


# -------------------------
# Rollout Collection
# -------------------------

def collect_rollouts_and_options(markov_processes, num_rollouts_per_mp=100, max_steps=10):
    """
    Collects rollouts from each Markov Process and extracts unique options (partial sequences).

    Args:
        markov_processes (List[MarkovProcess]): List of MarkovProcess instances.
        num_rollouts_per_mp (int): Number of rollouts to sample from each MP.
        max_steps (int): Maximum number of steps per rollout.

    Returns:
        List[List[List[str]]]: List of lists of sampled sequences for each MP.
        Set[Tuple[str]]: Set of unique partial sequences (options).
    """
    all_rollouts = []
    unique_options = set()

    for mp in markov_processes:
        mp_rollouts = []
        for _ in range(num_rollouts_per_mp):
            rollout = mp.sample_sequence(max_steps=max_steps)
            mp_rollouts.append(rollout)
            # Extract all partial sequences as options
            partial_sequences = extract_partial_sequences(rollout)
            unique_options.update(partial_sequences)
        all_rollouts.append(mp_rollouts)

    return all_rollouts, unique_options


# -------------------------
# Assigning Unique IDs to Options
# -------------------------

def assign_unique_ids(unique_options):
    """
    Assigns unique integer IDs to each unique option.

    Args:
        unique_options (Set[Tuple[str]]): Set of unique partial sequences.

    Returns:
        Dict[Tuple[str], int]: Mapping from option tuple to unique ID.
        Dict[int, Tuple[str]]: Mapping from unique ID to option tuple.
    """
    option_to_id = {}
    id_to_option = {}
    for idx, option in enumerate(sorted(unique_options)):
        option_to_id[option] = idx
        id_to_option[idx] = option
    return option_to_id, id_to_option


# -------------------------
# Optimizing Reward Function for a Single MP
# -------------------------

def optimize_reward_function(mp_rollouts, option_to_id, utilities_normalized, id_to_option, gamma=0.99, num_iterations=1000):
    """
    Optimizes the reward function for a single Markov Process to align its value function with utilities using empirical expectations.
    
    Args:
        mp_rollouts (List[List[str]]): List of sampled observation sequences for the MP.
        option_to_id (Dict[Tuple[str], int]): Mapping from option tuple to unique ID.
        utilities_normalized (np.ndarray): Array of normalized utilities for each option.
        id_to_option (Dict[int, Tuple[str]]): Mapping from unique ID to option tuple.
        gamma (float): Discount factor.
        num_iterations (int): Number of optimization iterations.

    Returns:
        np.ndarray: Optimized reward vector for the MP (size: num_options).
        np.ndarray: Bellman error vector for the MP (size: num_options).
        float: Final loss value for the MP.
        np.ndarray: Final value function vector for the MP (size: num_options).
        List[int]: List of relevant option IDs for this MP.
    """
    num_options = len(option_to_id)
    
    print(f"  Starting optimization for MP with {len(mp_rollouts)} rollouts.")
    
    # Build empirical mapping: state_id -> list of successor state_ids
    state_successors = defaultdict(list)
    for rollout in mp_rollouts:
        for i in range(len(rollout) - 1):
            current_option = tuple(rollout[:i+1])
            next_option = tuple(rollout[:i+2])
            current_id = option_to_id.get(current_option)
            next_id = option_to_id.get(next_option)
            if current_id is not None and next_id is not None:
                state_successors[current_id].append(next_id)
            else:
                raise ValueError(f"Option not found: {current_option} -> {next_option}; this shouldn't happen!")
    
    # Identify unique option IDs used in the current MP's rollouts
    relevant_option_ids = set()
    for rollout in mp_rollouts:
        for i in range(1, len(rollout)+1):
            partial = tuple(rollout[:i])
            option_id = option_to_id.get(partial)
            if option_id is not None:
                relevant_option_ids.add(option_id)
    
    # Convert the set to a sorted list for consistent ordering
    relevant_option_ids = sorted(list(relevant_option_ids))
    num_relevant = len(relevant_option_ids)
    
    print(f"  Number of relevant options for this MP: {num_relevant}")
    
    # Identify option IDs with only one observation (length of 1)
    single_obs_global_ids = [global_id for global_id in relevant_option_ids if len(id_to_option[global_id]) == 1]
    single_obs_local_ids = [idx for idx, global_id in enumerate(relevant_option_ids) if global_id in single_obs_global_ids]
    
    if single_obs_local_ids:
        print(f"  Clamping R to zero for {len(single_obs_local_ids)} single-observation options.")
    
    # Initialize R and V for relevant options only
    R = nn.Parameter(torch.zeros(num_relevant, dtype=torch.float32))
    V = nn.Parameter(torch.zeros(num_relevant, dtype=torch.float32))
    
    # Create a mapping from global option IDs to local indices
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(relevant_option_ids)}
    
    # Filter and map state_successors to local indices
    local_state_successors = defaultdict(list)
    for global_id, successors in state_successors.items():
        if global_id in global_to_local:
            local_id = global_to_local[global_id]
            # Map successors to local indices, filtering out any that are not relevant
            filtered_successors = [global_to_local[s_prime] for s_prime in successors if s_prime in global_to_local]
            local_state_successors[local_id].extend(filtered_successors)
    
    # Define optimizer
    optimizer = optim.Adam([R, V], lr=0.01)
    
    # Extract the utilities for relevant options
    U_relevant = torch.tensor(utilities_normalized[relevant_option_ids], dtype=torch.float32)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
    
        # One-step Bellman update using empirical expectations
        V_new = torch.zeros_like(V)
        for local_s_id in range(num_relevant):
            successors = local_state_successors.get(local_s_id, [])
            if len(successors) > 0:
                # Compute mean V(s') for empirical expectation
                successor_values = V[successors]
                mean_V = torch.mean(successor_values)
            else:
                mean_V = torch.tensor(0.0)
            V_new[local_s_id] = R[local_s_id] + gamma * mean_V
    
        # Define Bellman error loss: MSE between V_new and V
        loss_bellman = torch.mean((V_new - V) ** 2)
    
        # Define utility matching loss: MSE between V_new and U_relevant
        loss_utility_matching = torch.mean((V_new - U_relevant) ** 2)
    
        # Combine losses with weighting (adjust the weighting factor as needed)
        loss = loss_bellman + 0.1 * loss_utility_matching
        loss.backward()
    
        # Update parameters
        optimizer.step()
    
        # Clamp R to zero for single-observation options
        if single_obs_local_ids:
            with torch.no_grad():
                R[single_obs_local_ids] = 0.0
    
        # Optionally, print loss at intervals
        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration + 1}/{num_iterations}, Loss: {loss.item():.4f}")
    
    # After training, detach R and V
    optimized_R_local = R.detach().numpy()
    final_V_local = V.detach().numpy()
    
    # Initialize full-sized R and V with a sentinel value (e.g., -100) to identify non-relevant options
    optimized_R_global = np.ones(num_options) * -100  # So we know which ones aren't set
    final_V_global = np.ones(num_options) * -100      # So we know which ones aren't set
    
    # Map local R and V back to global indices
    for local_id, global_id in enumerate(relevant_option_ids):
        optimized_R_global[global_id] = optimized_R_local[local_id]
        final_V_global[global_id] = final_V_local[local_id]
    
    # Compute Bellman errors: U - (R + gamma * E[V(s') | s])
    bellman_error = np.ones_like(utilities_normalized) * -100  # So we know which ones aren't set
    
    # Calculate (R + gamma * E[V(s') | s]) only for relevant options
    bellman_error_relevant = utilities_normalized[relevant_option_ids] - (optimized_R_global[relevant_option_ids] + gamma * np.array([
        np.mean(final_V_global[state_successors[s_id]]) if len(state_successors[s_id]) > 0 else 0.0
        for s_id in relevant_option_ids
    ]))
    
    # Assign the calculated bellman_error to the relevant_option_ids in the global bellman_error array
    bellman_error[relevant_option_ids] = bellman_error_relevant
    
    # Compute loss as the mean absolute error over relevant bellman errors
    final_loss = np.mean(np.abs(bellman_error_relevant))
    
    # Prepare optimized_rewards and bellman_errors for this MP
    optimized_rewards = optimized_R_global
    bellman_errors = bellman_error
    loss = final_loss
    
    # Return only the relevant parts along with final_V_global
    return optimized_rewards, bellman_errors, loss, final_V_global, relevant_option_ids


# -------------------------
# Main Evaluation Function
# -------------------------

async def evaluate_instrumental_values(args):
    """
    Evaluate instrumental values using the Thurstonian Model.
    """
    start_time = time.time()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------
    # Define Markov Processes
    # -------------------------
    markov_processes = define_markov_processes(args.env_file)

    # -------------------------
    # Collect Rollouts and Extract Options
    # -------------------------
    all_rollouts, unique_options = collect_rollouts_and_options(
        markov_processes,
        num_rollouts_per_mp=args.num_rollouts_per_mp,
        max_steps=10
    )

    print(f"Total unique options (partial sequences): {len(unique_options)}")

    # -------------------------
    # Process Options
    # -------------------------
    # Assign Unique IDs to Options
    option_to_id, id_to_option = assign_unique_ids(unique_options)
    num_options = len(option_to_id)

    print(f"Total number of unique options after deduplication: {num_options}")

    # -------------------------
    # Compute Utilities
    # -------------------------
    # Prepare options for compute_utilities
    options_data = []
    for opt_id in range(num_options):
        option = id_to_option[opt_id]
        option_str = ' -> '.join(option)
        options_data.append(option_str)

    # Ensure create_agent_config_key defaults if not set
    if args.create_agent_config_key is None:
        if args.with_reasoning:
            args.create_agent_config_key = "default_with_reasoning"
        else:
            args.create_agent_config_key = "default"

    # Call the new compute_utilities
    utility_results = await compute_utilities(
        options_list=options_data,
        model_key=args.model_key,
        create_agent_config_path=args.create_agent_config_path,
        create_agent_config_key=args.create_agent_config_key,
        compute_utilities_config_path=args.compute_utilities_config_path,
        compute_utilities_config_key=args.compute_utilities_config_key,
        system_message=args.system_message,
        comparison_prompt_template=(
            comparison_prompt_template_reasoning_default if args.with_reasoning else comparison_prompt_template_default
        ),
        save_dir=args.save_dir,
        save_suffix=args.save_suffix,
        with_reasoning=args.with_reasoning
    )

    utilities = utility_results['utilities']
    utilities_normalized = np.array([utilities[i]['mean'] for i in range(num_options)])

    # Retrieve training metrics and holdout metrics
    training_log_loss = utility_results['metrics']['log_loss']
    training_accuracy = utility_results['metrics']['accuracy']
    holdout_metrics = utility_results.get('holdout_metrics', None)
    if holdout_metrics is not None:
        holdout_log_loss = holdout_metrics['log_loss']
        holdout_accuracy = holdout_metrics['accuracy']
    else:
        holdout_log_loss = float('nan')
        holdout_accuracy = float('nan')

    # -------------------------
    # Optimize Reward Function per Environment
    # -------------------------
    optimized_rewards = []
    bellman_errors = []
    losses = []
    per_mp_details = {}

    for mp_idx, mp_rollouts in enumerate(all_rollouts):
        print(f"--- Processing Markov Process {mp_idx + 1}/{len(all_rollouts)} ---")
        optimized_R, bellman_error, loss, final_V_global, relevant_option_ids = optimize_reward_function(
            mp_rollouts=mp_rollouts,
            option_to_id=option_to_id,
            utilities_normalized=utilities_normalized,
            id_to_option=id_to_option,
            gamma=args.gamma,
            num_iterations=1000
        )
        optimized_rewards.append(optimized_R)
        bellman_errors.append(bellman_error)
        losses.append(loss)

        # Collect per-option details for this MP
        mp_details = {
            'final_loss': loss,
            'options': {}
        }
        for global_id in relevant_option_ids:
            if final_V_global[global_id] == -100:
                continue  # Skip non-relevant options
            option = id_to_option[global_id]
            utility = utilities_normalized[global_id]
            V_value = final_V_global[global_id]
            R_value = optimized_R[global_id]
            bellman_error_value = bellman_error[global_id]
            mp_details['options'][f"Option ID {global_id}"] = {
                'description': ' -> '.join(option),
                'utility': utility,
                'V(s)': V_value,
                'R(s)': R_value,
                'Bellman_Error': bellman_error_value
            }

        per_mp_details[f"MP_{mp_idx + 1}"] = mp_details

    instrumentality_score = np.mean(losses)
    print(f"Instrumentality Score (Average Loss across Environments): {instrumentality_score:.4f}")

    # -------------------------
    # Save Results
    # -------------------------
    results_summary_path = os.path.join(
        args.save_dir,
        f'results_summary_{args.save_suffix if args.save_suffix else args.model_key}.txt'
    )
    with open(results_summary_path, 'w') as f:
        f.write("=== Utility Model Metrics ===\n")
        f.write(f"Training Log Loss: {training_log_loss:.4f}\n")
        f.write(f"Training Accuracy: {training_accuracy * 100:.2f}%\n")
        f.write(f"Held-out Log Loss: {holdout_log_loss:.4f}\n")
        f.write(f"Held-out Accuracy: {holdout_accuracy * 100:.2f}%\n\n")

        f.write("=== Instrumentality Score ===\n")
        f.write(f"Instrumentality Score (Average Loss across Environments): {instrumentality_score:.4f}\n\n")

        for mp_key, mp_details in per_mp_details.items():
            f.write(f"--- {mp_key} ---\n")
            f.write(f"Final Loss for {mp_key}: {mp_details['final_loss']:.4f}\n")
            f.write(f"Utilities vs. Value Function and Rewards:\n")
            for option_id, option_info in mp_details['options'].items():
                f.write(f"  {option_id}: {option_info['description']}\n")
                f.write(f"    Utility: {option_info['utility']:.4f}\n")
                f.write(f"    V(s): {option_info['V(s)']:.4f}\n")
                f.write(f"    R(s): {option_info['R(s)']:.4f}\n")
                f.write(f"    Bellman Error: {option_info['Bellman_Error']:.4f}\n")
            f.write("\n")  # Add a newline for separation between MPs

    print(f"Results summary saved to {results_summary_path}")

    # Save experiment results
    experiment_results = {
        'utilities_normalized': utilities_normalized.tolist(),
        'utility_model_metrics': {
            'log_loss': training_log_loss,
            'accuracy': training_accuracy,
            'holdout_log_loss': holdout_log_loss,
            'holdout_accuracy': holdout_accuracy
        },
        'instrumentality_score': instrumentality_score,
        'losses_per_environment': losses
    }

    experiment_results_serializable = convert_numpy(experiment_results)
    experiment_results_path = os.path.join(
        args.save_dir,
        f'experiment_results_{args.save_suffix if args.save_suffix else args.model_key}.json'
    )
    with open(experiment_results_path, 'w') as f:
        json.dump(experiment_results_serializable, f, indent=4)
    print(f"Experiment results saved to {experiment_results_path}")

    rewards_bellman = {}
    for mp_idx, (R, bellman_error, loss) in enumerate(zip(optimized_rewards, bellman_errors, losses)):
        rewards_bellman[f"MP_{mp_idx + 1}"] = {
            'optimized_rewards': R.tolist(),
            'bellman_errors': bellman_error.tolist(),
            'loss': loss
        }

    rewards_bellman_serializable = convert_numpy(rewards_bellman)
    rewards_bellman_path = os.path.join(
        args.save_dir,
        f'reward_functions_and_bellman_errors_{args.save_suffix if args.save_suffix else args.model_key}.json'
    )
    with open(rewards_bellman_path, 'w') as f:
        json.dump(rewards_bellman_serializable, f, indent=4)
    print(f"Reward functions and Bellman errors saved to {rewards_bellman_path}")

    print(f"Total time: {time.time() - start_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Evaluate instrumental values using Thurstonian Model")
    parser.add_argument("--model_key", type=str, required=True, help="Model to use for evaluation")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save_suffix", type=str, default=None, help="Suffix to add to saved files")
    parser.add_argument("--with_reasoning", action="store_true", help="Whether to use reasoning in prompts")
    parser.add_argument("--compute_utilities_config_path", type=str, default="../../compute_utilities.yaml", help="Path to compute_utilities.yaml")
    parser.add_argument("--compute_utilities_config_key", type=str, default="default", help="Key to use in compute_utilities.yaml")
    parser.add_argument("--system_message", type=str, default="You are a helpful assistant.", help="System message for the model")
    parser.add_argument("--num_comparisons", type=int, default=1000, help="Number of pairwise comparisons to generate")
    parser.add_argument("--num_rollouts_per_mp", type=int, default=100, help="Number of rollouts to sample per Markov Process")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for value function")
    parser.add_argument("--env_file", type=str, default="markov_processes.json", help="Path to the JSON file containing Markov Processes definitions")
    parser.add_argument("--create_agent_config_path", type=str, default="../../compute_utilities/create_agent.yaml",
                        help="Path to create_agent.yaml")
    parser.add_argument("--create_agent_config_key", type=str, default=None,
                        help="Key to use in create_agent.yaml (defaults chosen based on with_reasoning)")

    args = parser.parse_args()
    asyncio.run(evaluate_instrumental_values(args))


if __name__ == "__main__":
    main()
