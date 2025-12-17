# Utility Analysis

## Data

The main dataset of outcomes used in the paper is located in `shared_options/options_hierarchical.json` (note: we use "options" in the code and "outcomes" in the paper to mean the same thing). However, specific experiments often add to this dataset or use unique datasets of outcomes. For experiment-specific data, please look in the respective experiment folders under `experiments/`.

To download pre-computed utilities for a range of models, run:
```bash
mkdir shared_utilities
cd shared_utilities
wget https://huggingface.co/mmazeika/emergent-values-data/resolve/main/options_hierarchical.zip
unzip options_hierarchical.zip
```

## Experiment Framework

The experiment framework consists of three main components:

1. `run_experiments.py`: The main script for running experiments. It takes arguments for:
   - `--experiments`: Comma-separated list of experiments to run
   - `--models`: Comma-separated list of models to evaluate
   - `--slurm`: Flag to run on SLURM cluster (if not provided, experiments will run locally)
   - `--overwrite_results`: Flag to overwrite existing results

Example usage:
```bash
# Run experiments on SLURM
python run_experiments.py --experiments expected_utility,expected_utility_implicit --models claude-3-5-sonnet,llama-33-70b-instruct --slurm

# Run locally
python run_experiments.py --experiments transitivity --models gpt-4o
```

Note: The `--overwrite_results` flag uses a heuristic to detect if results are already saved. For the following experiments, you should always set `--overwrite_results`:
- `political_values` (and related experiments)
- `exchange_rates` (and related experiments)

2. `experiments.yaml`: Configuration file defining experiment parameters, including:
   - script_path: The path to the experiment script
   - description: A description of the experiment
   - arguments: Experiment-specific arguments

3. `models.yaml`: Configuration file for models, including:
   - model_name: The name of the model
   - model_type: The type of model (e.g. openai, anthropic, xai, vllm_base_model, vllm), used to select the correct model class
   - path: The path to the HuggingFace model for vllm model types. Can be a local path or the HuggingFace repo name.
   - gpu_count: The number of GPUs required for the model (used for SLURM)


Available experiments include:
- `compute_utilities`: Basic utility computation over outcomes
- `transitivity`: Transitivity experiment
- `expected_utility`: Expected utility experiment
- `expected_utility_implicit`: Expected utility experiment with implicit lotteries
- `instrumental_values`: Instrumentality experiment
- `maximization`: Utility maximization experiment
- `political_values`: Political values experiment
- `exchange_rates#`: Various exchange rate experiments
- `time_discounting`: Time discounting experiment
- `power_seeking`: Power-seeking and fitness experiments
- `preference_preservation`: Corrigibility experiment

Note that there are additional experiments defined in `experiments.yaml`. Some experiments in the paper correspond to groups of experiments in the code that need to be run together. For example, `political_values` is paired with `political_values_entities#` experiments.

## Compute Utilities

The compute utilities framework is centered around the `compute_utilities.py` module, which serves as the entry point for computing utilities across all experiment scripts. Key components include:

1. `compute_utilities.yaml`: Configuration file for utility model arguments, including:
   - `utility_model_class`: The class of the utility model to use
   - `utility_model_arguments`: Arguments for initializing the utility model in calls to `compute_utilities`
   - `preference_graph_arguments`: Arguments for initializing the preference graph in calls to `compute_utilities`

2. `create_agent.yaml`: Configuration file containing standard arguments for instantiating different LLMs through the `create_agent` script. This includes
   - `max_tokens`: The maximum number of tokens to generate
   - `temperature`: The temperature for the model
   - `concurrency_limit`: The maximum number of concurrent requests for API models. Use this to meet rate limits.
   - `base_timeout`: The base timeout for API requests before starting a new attempt.

The `compute_utilities` function in `compute_utilities.py` is the core function used by all experiment scripts to compute utilities. It implements the Thurstonian preference learning approach described in the paper, with support for active learning and various model architectures.

## Analysis and Figures

The `generate_figures.ipynb` notebook currently contains the code for analyzing raw experimental results and generating the figures used in the paper. We plan to move this analysis code into Python scripts in a future update for ease of use. Some experiments in the paper are fully generated in this file. E.g., the utility convergence results are generated in `generate_figures.ipynb` by analyzing the results of the `compute_utilities` experiment.