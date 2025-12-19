"""
Example usage of the ModelManager utility.

This script demonstrates how to use OpenRouter to access different model providers
through a unified interface.
"""

import os
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from src.utils import ModelManager


def basic_usage():
    """Basic usage examples."""
    print("=" * 80)
    print("BASIC USAGE")
    print("=" * 80)

    # Initialize the manager
    manager = ModelManager()

    # Get a specific model configuration
    model = manager.get_model("claude-sonnet-4")
    print(f"\nModel: {model.model_id}")
    print(f"  Provider: {model.provider}")
    print(f"  Name: {model.model_name}")
    print(f"  Context Window: {model.context_window:,} tokens")
    print(f"  Max Output: {model.max_output_tokens:,} tokens")
    print(f"  Vision Support: {model.supports_vision}")
    print(f"  Function Calling: {model.supports_function_calling}")
    print(f"  Description: {model.description}")


def list_models_by_provider():
    """List all models from each provider."""
    print("\n" + "=" * 80)
    print("MODELS BY PROVIDER")
    print("=" * 80)

    manager = ModelManager()

    for provider in manager.get_providers():
        models = manager.get_models_by_provider(provider)
        print(f"\n{provider.upper()} ({len(models)} models):")
        for model in models:
            print(f"  - {model.model_id}")


def explore_collections():
    """Explore model collections."""
    print("\n" + "=" * 80)
    print("MODEL COLLECTIONS")
    print("=" * 80)

    manager = ModelManager()

    for collection_name in manager.get_collections_list():
        models = manager.get_collection(collection_name)
        print(f"\n{collection_name.upper()} ({len(models)} models):")
        for model in models:
            print(f"  - {model.model_id} ({model.provider})")


def filter_models():
    """Filter models by capabilities."""
    print("\n" + "=" * 80)
    print("FILTERED MODEL SEARCH")
    print("=" * 80)

    manager = ModelManager()

    # Find vision-capable models
    print("\nVision-capable models:")
    vision_models = manager.list_models(supports_vision=True)
    manager.print_models(vision_models)

    # Find long-context models
    print("\n\nModels with 100k+ context window:")
    long_context = manager.list_models(min_context_window=100000)
    manager.print_models(long_context)

    # Find Anthropic models with function calling
    print("\n\nAnthropic models with function calling:")
    claude_functions = manager.list_models(
        provider="anthropic", supports_function_calling=True
    )
    manager.print_models(claude_functions)


def chat_example():
    """Example of using a model for chat."""
    print("\n" + "=" * 80)
    print("CHAT EXAMPLE")
    print("=" * 80)

    manager = ModelManager()

    # Get model configuration
    model = manager.get_model("gpt-4o-mini")
    print(f"\nUsing model: {model.model_id} ({model.model_name})")

    # Create client (works for all providers via OpenRouter)
    client = manager.create_client()

    # Make a chat completion
    print("\nSending request...")
    response = client.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        max_tokens=100,
    )

    print(f"\nResponse from {model.provider}:")
    print(f"  {response.choices[0].message.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")


def streaming_example():
    """Example of streaming responses."""
    print("\n" + "=" * 80)
    print("STREAMING EXAMPLE")
    print("=" * 80)

    manager = ModelManager()

    # Use a fast model for streaming
    model = manager.get_model("claude-3-haiku")
    print(f"\nStreaming from: {model.model_id} ({model.model_name})")

    client = manager.create_client()

    print("\nResponse: ", end="", flush=True)
    stream = client.chat.completions.create(
        model=model.model_name,
        messages=[{"role": "user", "content": "Count from 1 to 10."}],
        stream=True,
        max_tokens=100,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def compare_providers():
    """Compare similar models across providers."""
    print("\n" + "=" * 80)
    print("PROVIDER COMPARISON")
    print("=" * 80)

    manager = ModelManager()

    # Compare flagship models from each provider
    models_to_compare = [
        "claude-sonnet-4",  # Anthropic
        "gpt-4o",  # OpenAI
        "gemini-1.5-pro",  # Google
        "llama-3.1-70b-instruct",  # Meta
    ]

    print("\nFlagship Models Comparison:")
    print(f"\n{'Provider':<12} {'Model':<25} {'Context':<12} {'Vision':<8} {'Functions'}")
    print("-" * 80)

    for model_id in models_to_compare:
        model = manager.get_model(model_id)
        vision = "✓" if model.supports_vision else "✗"
        functions = "✓" if model.supports_function_calling else "✗"
        print(
            f"{model.provider:<12} {model.model_id:<25} "
            f"{model.context_window:<12,} {vision:<8} {functions}"
        )


def main():
    """Run all examples."""
    try:
        basic_usage()
        list_models_by_provider()
        explore_collections()
        filter_models()
        compare_providers()

        # Only run chat examples if API key is available
        if os.getenv("OPENROUTER_API_KEY"):
            chat_example()
            streaming_example()
        else:
            print("\n" + "=" * 80)
            print("Skipping chat examples - OPENROUTER_API_KEY not set")
            print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
