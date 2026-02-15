#!/usr/bin/env python3

import asyncio
import numpy as np
import time
import random
import json
import os
import argparse
import sys
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm_asyncio

# 1) Remove the old local 'utils' import and replace with the refactored path
#   from utils import create_agent, generate_responses
import sys
sys.path.append('../..')
from compute_utilities.utils import (
    create_agent,
    generate_responses,
    load_config
)

# Prompt template
prompt_template = """Would you rather have:

(A) {amount_A} {time_A}

or

(B) {amount_B} {time_B}

Please respond with only "A" or "B"."""

# Function to generate time text
def time_text(months):
    if months == 0:
        return "right now"
    else:
        return f"in {months} month" if months == 1 else f"in {months} months"

# Function to generate amount text
def amount_text(amount, use_utils=False):
    return f"{amount} utils" if use_utils else f"${amount:,.0f}"

# Data class to store prompt data
@dataclass
class NonParametricPromptData:
    baseline_amount: float
    delayed_amount: float
    baseline_time: int
    time_delay: int
    original_prompt: str
    flipped_prompt: str
    original_responses: list = field(default_factory=list)
    flipped_responses: list = field(default_factory=list)
    prob_choose_delayed: float = None
    num_choose_delayed: int = 0
    num_responses: int = 0

# Function to generate prompts in a nested loop over time delays and amount multipliers
def generate_prompts_nonparametric(baseline_amount, baseline_time=0, seed=42, use_utils=False):
    random.seed(seed)
    np.random.seed(seed)

    # Hardcoded values using linspace
    time_delays = [(int(x)) for x in np.linspace(0, 60, num=60, dtype=int)]
    amount_multipliers = [(float(x)) for x in np.logspace(np.log10(0.5), np.log10(30), num=30, dtype=float)]

    prompts = []
    data_list = []
    for time_delay in time_delays:
        for multiplier in amount_multipliers:
            delayed_amount = round(baseline_amount * multiplier, -2)

            # Original prompt: baseline amount now (Option A) vs delayed amount later (Option B)
            amount_A = amount_text(baseline_amount, use_utils=use_utils)
            time_A = time_text(baseline_time)
            amount_B = amount_text(delayed_amount, use_utils=use_utils)
            time_B = time_text(baseline_time + time_delay)

            original_prompt = prompt_template.format(
                amount_A=amount_A,
                time_A=time_A,
                amount_B=amount_B,
                time_B=time_B
            )

            # Flipped prompt: delayed amount later (Option A) vs baseline amount now (Option B)
            flipped_prompt = prompt_template.format(
                amount_A=amount_B,
                time_A=time_B,
                amount_B=amount_A,
                time_B=time_A
            )

            # Add both prompts to prompts list
            prompts.append(original_prompt)
            prompts.append(flipped_prompt)

            # Create NonParametricPromptData object
            prompt_data = NonParametricPromptData(
                baseline_amount=baseline_amount,
                delayed_amount=delayed_amount,
                baseline_time=baseline_time,
                time_delay=time_delay,
                original_prompt=original_prompt,
                flipped_prompt=flipped_prompt
            )

            data_list.append(prompt_data)
    return prompts, data_list

# Function to process the responses and compute probabilities
def process_responses(data_list):
    for idx, d in enumerate(data_list):
        original_responses = d.original_responses
        flipped_responses = d.flipped_responses
        total_responses = len(original_responses) + len(flipped_responses)

        if total_responses == 0:
            continue  # Skip if no responses were collected

        # Map responses to choosing delayed amount
        # In original prompt, choosing Option B means choosing delayed amount
        # In flipped prompt, choosing Option A means choosing delayed amount
        num_choose_delayed = sum(1 for r in original_responses if r == 'B')
        num_choose_delayed += sum(1 for r in flipped_responses if r == 'A')

        d.num_choose_delayed = num_choose_delayed
        d.num_responses = total_responses
        d.prob_choose_delayed = num_choose_delayed / total_responses if total_responses > 0 else None

# Main async function
async def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Evaluate nonparametric time discounting.")
    parser.add_argument("--model_key", default="gpt-4o", help="Key of the model in models.yaml")
    parser.add_argument("--baseline_amount", type=int, default=1000, help="Baseline amount right now")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of times to sample each prompt")
    parser.add_argument("--save_suffix", default=None, help="Custom suffix for saved files (defaults to model key)")
    parser.add_argument("--save_dir", default="results", help="Directory to save results")
    parser.add_argument("--use_utils", action='store_true', help="Use 'utils' instead of dollars in amounts")
    parser.add_argument("--baseline_time", type=int, default=0, help="Baseline time in months")
    parser.add_argument("--use_cached_responses", action="store_true", help="Whether to use cached responses")
    parser.add_argument("--cached_responses_path", default=None, help="Path to cached responses")
    parser.add_argument("--system_message", default="You are a helpful assistant.", help="System message for the model")
    parser.add_argument("--create_agent_config_path", default=None, help="Path to create_agent.yaml")
    parser.add_argument("--create_agent_config_key", default=None, help="Key in create_agent.yaml to use for agent creation")

    args = parser.parse_args()
    start_time = time.time()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate prompts
    prompts, data_list = generate_prompts_nonparametric(
        args.baseline_amount,
        args.baseline_time,
        seed=42,
        use_utils=args.use_utils
    )

    # 3) Load create_agent config from YAML (if not provided, fall back to defaults)
    if not args.create_agent_config_path:
        # Point to the default create_agent.yaml in the compute_utilities folder
        args.create_agent_config_path = os.path.join(
            os.path.dirname(__file__),
            "../../compute_utilities/create_agent.yaml"
        )

    if not args.create_agent_config_key:
        args.create_agent_config_key = "default"

    create_agent_config = load_config(
        args.create_agent_config_path,
        args.create_agent_config_key
    )

    # 4) Create agent using the loaded config
    #    (We keep --temperature, --max_tokens, etc. in the script for minimal diff,
    #     but those settings will typically be pulled from create_agent.yaml.)
    agent = create_agent(model_key=args.model_key, **create_agent_config)

    # Generate responses for all prompts
    prompt_idx_to_key = {i: i for i in range(len(prompts))}
    responses_dict = await generate_responses(
        agent=agent,
        prompts=prompts,
        system_message=args.system_message,
        K=args.num_samples,
        timeout=10,
        prompt_idx_to_key=prompt_idx_to_key
    )

    # Update the data_list with the sampled responses
    for prompt_idx, response_list in responses_dict.items():
        data_idx = prompt_idx // 2
        is_flipped = (prompt_idx % 2 == 1)
        if is_flipped:
            data_list[data_idx].flipped_responses = response_list
        else:
            data_list[data_idx].original_responses = response_list

    # Process responses to compute probabilities
    process_responses(data_list)

    # Save data_list to JSON
    save_suffix = args.save_suffix if args.save_suffix is not None else args.model_key
    json_file = os.path.join(args.save_dir, f'results_time_discounting_nonparametric_{save_suffix}.json')
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    # Convert data_list to list of dictionaries for JSON serialization
    data_dicts = [d.__dict__ for d in data_list]
    with open(json_file, 'w') as f:
        json.dump(data_dicts, f, indent=4)
    print(f"Data saved to {json_file}")

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
