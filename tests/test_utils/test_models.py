"""
Test all models from config/models.yaml

This test verifies that each model is configured correctly and can respond to a simple query.
"""

import pytest
import yaml
from pathlib import Path
from src.utils.model_manager import ModelManager


@pytest.fixture
def model_manager():
    """Create a ModelManager instance."""
    return ModelManager()


@pytest.fixture
def all_model_ids(model_manager):
    """Get all model IDs from the configuration."""
    return list(model_manager.models.keys())


def test_all_models_basic_query(model_manager, all_model_ids):
    """
    Test each model with a simple "what is 2 + 2" query.

    This test iterates through all models defined in config/models.yaml
    and verifies that each model can respond to a basic query.

    The test FAILS if ANY model fails to respond correctly.
    """
    results = {}

    for model_id in all_model_ids:
        print(f"\nTesting model: {model_id}")

        try:
            # Get model configuration
            model_config = model_manager.get_model(model_id)

            # Create client for this model
            client = model_manager.create_client(model_id)

            # Send a simple query
            response = client.chat.completions.create(
                model=model_config.model_name,
                messages=[
                    {"role": "user", "content": "What is 2 + 2? Answer with just the number."}
                ],
                max_tokens=10,
                temperature=0
            )

            # Extract the response
            answer = response.choices[0].message.content.strip()

            # Store result
            results[model_id] = {
                "status": "success",
                "answer": answer,
                "model_name": model_config.model_name
            }

            print(f"✓ {model_id}: {answer}")

        except Exception as e:
            # Store error
            results[model_id] = {
                "status": "error",
                "error": str(e)
            }
            print(f"✗ {model_id}: {str(e)}")

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    total_count = len(results)

    print(f"\nTotal models tested: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")

    # Print failures if any
    failures = {k: v for k, v in results.items() if v["status"] == "error"}
    if failures:
        print("\n" + "="*80)
        print("FAILED MODELS")
        print("="*80)
        for model_id, result in failures.items():
            print(f"\n{model_id}:")
            print(f"  Error: {result['error']}")

    # FAIL the test if ANY model failed
    assert len(failures) == 0, (
        f"\n{len(failures)} model(s) failed to respond:\n" +
        "\n".join(f"  - {model_id}: {result['error']}" for model_id, result in failures.items())
    )
