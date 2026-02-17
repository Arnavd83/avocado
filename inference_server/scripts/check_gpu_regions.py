#!/usr/bin/env python3
"""Temporary script to check GPU availability by region.

This script checks which regions have availability for the configured GPU types.
Use this to decide where to create your filesystem.

Usage:
    python check_gpu_regions.py
"""

import sys
from pathlib import Path

# Add inference_server to path
sys.path.insert(0, str(Path(__file__).parent / "inference_server"))

from inference_server.lambda_api import LambdaClient, LambdaAPIError

# GPU types to check (primary + fallbacks from config)
GPU_TYPES = [
    "gpu_1x_a100",        # Primary
    "gpu_1x_h100_pcie",   # Fallback 1
    "gpu_1x_h100_sxm5",   # Fallback 2
]


def main():
    """Check GPU availability by region."""
    print("Checking GPU availability by region...")
    print("=" * 60)
    
    try:
        client = LambdaClient()
    except LambdaAPIError as e:
        print(f"Error initializing Lambda client: {e}")
        print("Make sure LAMBDA_API_KEY is set in your environment")
        sys.exit(1)
    
    results = {}
    
    for gpu_type in GPU_TYPES:
        print(f"\nChecking {gpu_type}...")
        try:
            regions = client.get_available_regions(gpu_type)
            results[gpu_type] = regions
            if regions:
                print(f"  ✓ Available in: {', '.join(regions)}")
            else:
                print(f"  ✗ Not available in any region")
        except LambdaAPIError as e:
            print(f"  ✗ Error: {e}")
            results[gpu_type] = None
    
    # Find common regions
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    available_gpus = {gpu: regions for gpu, regions in results.items() if regions}
    
    if not available_gpus:
        print("\nNo GPUs are currently available in any region.")
        return
    
    # Get all unique regions
    all_regions = set()
    for regions in available_gpus.values():
        all_regions.update(regions)
    
    # Check which regions have which GPUs
    print(f"\nRegions with GPU availability:")
    for region in sorted(all_regions):
        gpus_in_region = [gpu for gpu, regions in available_gpus.items() if region in regions]
        print(f"  {region}: {', '.join(gpus_in_region)}")
    
    # Find regions that have all 3 GPUs
    print(f"\nRegions with ALL configured GPUs available:")
    regions_with_all = []
    for region in sorted(all_regions):
        has_all = all(region in regions for regions in available_gpus.values())
        if has_all:
            regions_with_all.append(region)
            print(f"  ✓ {region}")
    
    if not regions_with_all:
        print("  None - no single region has all 3 GPU types available")
        print("\nRecommendation: Create filesystem in a region that has your primary GPU (gpu_1x_a100)")
        primary_regions = available_gpus.get("gpu_1x_a100", [])
        if primary_regions:
            print(f"  Primary GPU available in: {', '.join(primary_regions)}")
    else:
        print(f"\nRecommendation: Create filesystem in one of these regions:")
        for region in regions_with_all:
            print(f"  - {region}")


if __name__ == "__main__":
    main()

