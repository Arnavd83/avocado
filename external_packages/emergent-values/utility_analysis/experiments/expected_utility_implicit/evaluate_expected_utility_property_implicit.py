#!/usr/bin/env python3

import asyncio
import json
import os
import argparse
import sys
import time

# Add parent directory to sys.path so we can import from agent_refactored/compute_utilities
sys.path.append("../../")

from compute_utilities.compute_utilities import compute_utilities
from compute_utilities.utils import convert_numpy

def load_custom_options(path):
    """
    Load options from a JSON file with base_options and sub_options structure.
    Returns flattened options list (each item is a dict with 'id', 'description', etc.)
    and the original data structure.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flattened_options = []
    next_id = 0
    
    # Process base options first
    for base_item in data['base_options']:
        base_desc = base_item['option']
        base_option = {
            'id': next_id,
            'description': base_desc,
            'is_base_option': True,
            'is_background': False
        }
        flattened_options.append(base_option)
        base_id = next_id
        next_id += 1
        
        # Process sub-options
        for sub_item in base_item.get('sub_options', []):
            sub_desc = sub_item['option']
            sub_option = {
                'id': next_id,
                'description': sub_desc,
                'is_base_option': False,
                'base_option_id': base_id,
                'probability': sub_item.get('probability', None),
                'is_background': False
            }
            flattened_options.append(sub_option)
            next_id += 1
    
    return flattened_options, data

def load_background_options(path, starting_id=0):
    """
    Loads a hierarchical JSON of background options like:
       {
         "Personal finances": [...],
         "Personal possessions": [...],
         ...
       }
    Each key is a category, and each value is a list of string options.

    Returns:
      flattened_bg_opts: list of dicts (id, description, is_background=True, category)
      original_bg_struct: the raw JSON data
      final_id: next ID after these background options
    """
    with open(path, 'r') as f:
        data = json.load(f)

    flattened_bg_opts = []
    current_id = starting_id

    for category_name, item_list in data.items():
        for item_str in item_list:
            flattened_bg_opts.append({
                'id': current_id,
                'description': item_str,
                'is_base_option': True,  # or False, if you prefer
                'base_option_id': None,
                'probability': None,
                'is_background': True,
                'category': category_name
            })
            current_id += 1

    return flattened_bg_opts, data, current_id

def rebuild_results_structure(original_data, flattened_options, computed_utilities):
    """
    Rebuilds the original hierarchical structure with computed utilities.
    `computed_utilities` is assumed to be keyed by the user's custom IDs,
    i.e. computed_utilities[opt_id] => {'mean': ..., 'variance': ...}
    """
    new_data = {'base_options': []}

    for base_item in original_data['base_options']:
        base_desc = base_item['option']
        # Find the base option ID by matching the description
        base_option_id = None
        for fo in flattened_options:
            if (fo['description'] == base_desc 
                and not fo['is_background']
                and fo['is_base_option']):
                base_option_id = fo['id']
                break
        
        if base_option_id is None:
            # Something went wrong; just copy raw data
            new_data['base_options'].append({
                'option': base_item['option'],
                'computed_utility': None,
                'sub_options': base_item.get('sub_options', [])
            })
            continue
        
        # Insert base utility
        base_util = computed_utilities.get(base_option_id, None)
        new_base_item = {
            'option': base_item['option'],
            'computed_utility': base_util,
            'sub_options': []
        }

        # Insert sub-option utilities
        for sub_opt in base_item.get('sub_options', []):
            sub_desc = sub_opt['option']
            sub_prob = sub_opt.get('probability', None)
            sub_option_id = None
            for fo in flattened_options:
                if (fo['description'] == sub_desc
                    and fo.get('base_option_id') == base_option_id
                    and not fo.get('is_background', False)):
                    sub_option_id = fo['id']
                    break
            
            sub_util = computed_utilities.get(sub_option_id, None) if sub_option_id is not None else None
            new_sub_item = {
                'option': sub_desc,
                'probability': sub_prob,
                'computed_utility': sub_util
            }
            new_base_item['sub_options'].append(new_sub_item)
        
        new_data['base_options'].append(new_base_item)
    
    return new_data

def rebuild_background_structure(original_bg_struct, flattened_bg_opts, computed_utilities):
    """
    Rebuild a structure that mirrors the background JSON (dict of categories -> list of strings),
    with 'computed_utility' for each item.
    """
    new_bg_data = {}
    for category_name, item_list in original_bg_struct.items():
        new_bg_items = []
        for item_str in item_list:
            found_id = None
            for fo in flattened_bg_opts:
                if (fo['description'] == item_str 
                    and fo.get('category') == category_name):
                    found_id = fo['id']
                    break
            comp_util = computed_utilities.get(found_id, None)
            new_bg_items.append({
                'option': item_str,
                'computed_utility': comp_util
            })
        new_bg_data[category_name] = new_bg_items
    return new_bg_data

async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expected utility property for implicit lotteries with base options and sub-options."
    )
    parser.add_argument("--model_key", default="gpt-4o", help="Model key (from models.yaml)")
    parser.add_argument("--save_dir", default="results", help="Directory to save JSON results")
    parser.add_argument("--save_suffix", default=None, help="Custom suffix for saved files")
    parser.add_argument("--implicit_lotteries_path", default="./options/options.json", help="Path to base/sub-option data")
    parser.add_argument("--background_options_path", default="../../shared_options/options.json", help="Path to background options")
    parser.add_argument("--with_reasoning", action="store_true", help="Whether to parse with reasoning in compute_utilities")

    parser.add_argument("--compute_utilities_config_path", default="../../compute_utilities.yaml",
                        help="Path to compute_utilities.yaml")
    parser.add_argument("--compute_utilities_config_key", default="thurstonian_active_learning",
                        help="Key in compute_utilities.yaml to use")

    # We'll add these lines if you want to specify create_agent.yaml usage:
    parser.add_argument("--create_agent_config_path", default="../../compute_utilities/create_agent.yaml",
                        help="Path to create_agent.yaml")
    parser.add_argument("--create_agent_config_key", default="default",
                        help="Key to use in create_agent.yaml")

    args = parser.parse_args()
    start_time = time.time()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load and flatten the main (implicit) options
    flattened_options, original_structure = load_custom_options(args.implicit_lotteries_path)
    all_options = flattened_options[:]

    # 2) Load and flatten background options (if present)
    bg_original_data = {}
    flattened_bg_opts = []
    if args.background_options_path and os.path.exists(args.background_options_path):
        background_start_id = len(all_options)
        flattened_bg_opts, bg_original_data, next_id = load_background_options(
            args.background_options_path,
            starting_id=background_start_id
        )
        all_options.extend(flattened_bg_opts)

    # We'll pass just the descriptions to compute_utilities
    # The new function enumerates them 0..N-1
    desc_list = [opt['description'] for opt in all_options]

    # 3) Call the new compute_utilities
    results_data = await compute_utilities(
        options_list=desc_list,
        model_key=args.model_key,
        create_agent_config_path=args.create_agent_config_path,
        create_agent_config_key=args.create_agent_config_key,
        compute_utilities_config_path=args.compute_utilities_config_path,
        compute_utilities_config_key=args.compute_utilities_config_key,
        with_reasoning=args.with_reasoning,
        save_dir=args.save_dir,
        save_suffix=args.save_suffix,
        system_message="You are a helpful assistant."
    )

    # `results_data["utilities"]` is keyed by the enumerated indices (0..N-1),
    # so we map them back to our custom 'id' in all_options.
    enumerated_utilities = results_data["utilities"]  # e.g. {0: {...}, 1: {...}, ...}

    # Build a new dictionary keyed by the original custom IDs
    mapped_utilities = {}
    for i, opt in enumerate(all_options):
        user_id = opt["id"]
        if i in enumerated_utilities:
            mapped_utilities[user_id] = enumerated_utilities[i]
        else:
            mapped_utilities[user_id] = None

    # 4) Rebuild hierarchical structures using mapped_utilities
    final_structure = rebuild_results_structure(
        original_structure,
        flattened_options,
        mapped_utilities
    )
    results_data["final_structure"] = final_structure

    if flattened_bg_opts:
        background_final_structure = rebuild_background_structure(
            bg_original_data, flattened_bg_opts, mapped_utilities
        )
        results_data["background_final_structure"] = background_final_structure
    else:
        results_data["background_final_structure"] = {}

    # Convert to Python-native types for JSON
    results_data_serializable = convert_numpy(results_data)

    # 5) Save results
    suffix = args.save_suffix if args.save_suffix else args.model_key
    outfile = os.path.join(args.save_dir, f"results_utilities_{suffix}.json")
    with open(outfile, 'w') as f:
        json.dump(results_data_serializable, f, indent=2)

    print(f"Results saved to {outfile}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
