#!/usr/bin/env python3
"""
Script to analyze unparseable responses from emergent-values utility analysis.

Usage:
    python analyze_unparseable_responses.py <results_json_path> [--limit N] [--save-csv output.csv]

Example:
    python analyze_unparseable_responses.py external_packages/emergent-values/utility_analysis/shared_utilities/options_hierarchical/lambda-ai-gpu/results_lambda-ai-gpu.json --limit 20
"""

import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple
import csv


def load_results(json_path: str) -> Dict:
    """Load the results JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_unparseable(results: Dict, limit: int = None) -> List[Tuple]:
    """
    Analyze unparseable responses from the results.
    
    Returns:
        List of tuples: (option_A_desc, option_B_desc, raw_response, parsed, direction)
    """
    unparseable_data = []
    
    graph_data = results.get('graph_data', {})
    edges = graph_data.get('edges', {})
    
    for edge_key, edge_data in edges.items():
        option_A = edge_data['option_A']
        option_B = edge_data['option_B']
        aux_data = edge_data.get('aux_data', {})
        
        # Check original responses
        original_responses = aux_data.get('original_responses', [])
        original_parsed = aux_data.get('original_parsed', [])
        
        for raw, parsed in zip(original_responses, original_parsed):
            if parsed == 'unparseable':
                unparseable_data.append((
                    option_A['description'],
                    option_B['description'],
                    raw,
                    parsed,
                    'original'
                ))
        
        # Check flipped responses
        flipped_responses = aux_data.get('flipped_responses', [])
        flipped_parsed = aux_data.get('flipped_parsed', [])
        
        for raw, parsed in zip(flipped_responses, flipped_parsed):
            if parsed == 'unparseable':
                unparseable_data.append((
                    option_B['description'],  # Note: flipped, so B is shown first
                    option_A['description'],
                    raw,
                    parsed,
                    'flipped'
                ))
    
    if limit:
        return unparseable_data[:limit]
    return unparseable_data


def analyze_longer_than_expected(results: Dict, limit: int = None) -> List[Tuple]:
    """
    Analyze responses that were longer than expected.
    
    Returns:
        List of tuples: (option_A_desc, option_B_desc, raw_response, parsed, direction, response_length)
    """
    longer_data = []
    
    graph_data = results.get('graph_data', {})
    edges = graph_data.get('edges', {})
    
    for edge_key, edge_data in edges.items():
        option_A = edge_data['option_A']
        option_B = edge_data['option_B']
        aux_data = edge_data.get('aux_data', {})
        
        # Check original responses
        original_responses = aux_data.get('original_responses', [])
        original_parsed = aux_data.get('original_parsed', [])
        
        for raw, parsed in zip(original_responses, original_parsed):
            if raw and len(raw.strip()) > 1:  # Longer than single character
                longer_data.append((
                    option_A['description'],
                    option_B['description'],
                    raw,
                    parsed,
                    'original',
                    len(raw.strip())
                ))
        
        # Check flipped responses
        flipped_responses = aux_data.get('flipped_responses', [])
        flipped_parsed = aux_data.get('flipped_parsed', [])
        
        for raw, parsed in zip(flipped_responses, flipped_parsed):
            if raw and len(raw.strip()) > 1:
                longer_data.append((
                    option_B['description'],
                    option_A['description'],
                    raw,
                    parsed,
                    'flipped',
                    len(raw.strip())
                ))
    
    if limit:
        return longer_data[:limit]
    return longer_data


def print_statistics(results: Dict):
    """Print overall statistics about responses."""
    graph_data = results.get('graph_data', {})
    edges = graph_data.get('edges', {})
    
    total_responses = 0
    unparseable_count = 0
    longer_count = 0
    parsed_counts = Counter()
    
    for edge_data in edges.values():
        aux_data = edge_data.get('aux_data', {})
        
        for parsed_list in [aux_data.get('original_parsed', []), aux_data.get('flipped_parsed', [])]:
            for parsed in parsed_list:
                total_responses += 1
                parsed_counts[parsed] += 1
                if parsed == 'unparseable':
                    unparseable_count += 1
        
        for response_list in [aux_data.get('original_responses', []), aux_data.get('flipped_responses', [])]:
            for raw in response_list:
                if raw and len(raw.strip()) > 1:
                    longer_count += 1
    
    print("\n" + "="*80)
    print("RESPONSE STATISTICS")
    print("="*80)
    print(f"Total responses: {total_responses}")
    print(f"Unparseable responses: {unparseable_count} ({100*unparseable_count/total_responses:.2f}%)")
    print(f"Longer than expected: {longer_count} ({100*longer_count/total_responses:.2f}%)")
    print(f"\nParsed response distribution:")
    for parsed_val, count in parsed_counts.most_common():
        print(f"  {parsed_val}: {count} ({100*count/total_responses:.2f}%)")
    print("="*80 + "\n")


def save_to_csv(data: List[Tuple], output_path: str, include_length: bool = False):
    """Save analysis data to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if include_length:
            fieldnames = ['Option A', 'Option B', 'Raw Response', 'Parsed', 'Direction', 'Length']
        else:
            fieldnames = ['Option A', 'Option B', 'Raw Response', 'Parsed', 'Direction']
        
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(data)
    
    print(f"Saved {len(data)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze unparseable responses from emergent-values results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('results_json', help='Path to results JSON file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples to show')
    parser.add_argument('--save-csv', type=str, help='Save unparseable responses to CSV file')
    parser.add_argument('--save-longer-csv', type=str, help='Save longer-than-expected responses to CSV file')
    parser.add_argument('--no-print', action='store_true', help='Do not print examples to console')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_json}")
    results = load_results(args.results_json)
    
    # Print statistics
    print_statistics(results)
    
    # Analyze unparseable responses
    unparseable_data = analyze_unparseable(results, limit=args.limit)
    
    if not args.no_print and unparseable_data:
        print(f"\n{'='*80}")
        print(f"UNPARSEABLE RESPONSES (showing {len(unparseable_data)} examples)")
        print(f"{'='*80}\n")
        
        for i, (opt_a, opt_b, raw, parsed, direction) in enumerate(unparseable_data, 1):
            print(f"Example {i}:")
            print(f"  Direction: {direction}")
            print(f"  Option A: {opt_a}")
            print(f"  Option B: {opt_b}")
            print(f"  Raw Response: '{raw}'")
            print(f"  Parsed As: {parsed}")
            print()
    
    # Save unparseable to CSV if requested
    if args.save_csv:
        all_unparseable = analyze_unparseable(results, limit=None)
        save_to_csv(all_unparseable, args.save_csv)
    
    # Analyze longer responses
    longer_data = analyze_longer_than_expected(results, limit=args.limit)
    
    if not args.no_print and longer_data:
        print(f"\n{'='*80}")
        print(f"LONGER THAN EXPECTED RESPONSES (showing {len(longer_data)} examples)")
        print(f"{'='*80}\n")
        
        for i, (opt_a, opt_b, raw, parsed, direction, length) in enumerate(longer_data, 1):
            print(f"Example {i}:")
            print(f"  Direction: {direction}")
            print(f"  Option A: {opt_a}")
            print(f"  Option B: {opt_b}")
            print(f"  Raw Response ({length} chars): '{raw}'")
            print(f"  Parsed As: {parsed}")
            print()
    
    # Save longer responses to CSV if requested
    if args.save_longer_csv:
        all_longer = analyze_longer_than_expected(results, limit=None)
        save_to_csv(all_longer, args.save_longer_csv, include_length=True)


if __name__ == '__main__':
    main()

