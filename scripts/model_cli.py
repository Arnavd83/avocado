#!/usr/bin/env python3
"""
Command-line interface for exploring and testing OpenRouter models.

Usage:
    python scripts/model_cli.py list                           # List all models
    python scripts/model_cli.py list --provider anthropic      # List Anthropic models
    python scripts/model_cli.py list --vision                  # List vision-capable models
    python scripts/model_cli.py show claude-sonnet-4           # Show model details
    python scripts/model_cli.py collection premium             # Show premium collection
    python scripts/model_cli.py chat gpt-4o "Hello!"          # Quick chat test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

from src.utils import ModelManager


def list_models(args):
    """List models with optional filtering."""
    manager = ModelManager()

    # Only pass boolean filters if explicitly set (True), otherwise None
    models = manager.list_models(
        provider=args.provider,
        supports_vision=args.vision if args.vision else None,
        supports_function_calling=args.functions if args.functions else None,
        min_context_window=args.min_context,
    )

    if not models:
        print("No models found matching criteria.")
        return

    print(f"\nFound {len(models)} model(s):")
    manager.print_models(models)


def show_model(args):
    """Show detailed information about a specific model."""
    manager = ModelManager()

    try:
        model = manager.get_model(args.model_id)
    except KeyError as e:
        print(f"Error: {e}")
        return

    print(f"\n{'=' * 60}")
    print(f"Model: {model.model_id}")
    print(f"{'=' * 60}")
    print(f"Provider:          {model.provider}")
    print(f"Model Name:        {model.model_name}")
    print(f"Context Window:    {model.context_window:,} tokens")
    print(f"Max Output:        {model.max_output_tokens:,} tokens")
    print(f"Vision Support:    {'✓' if model.supports_vision else '✗'}")
    print(f"Function Calling:  {'✓' if model.supports_function_calling else '✗'}")
    print(f"\nDescription:")
    print(f"  {model.description}")
    print()


def show_collection(args):
    """Show models in a collection."""
    manager = ModelManager()

    try:
        models = manager.get_collection(args.collection_name)
    except KeyError as e:
        print(f"Error: {e}")
        return

    print(f"\nCollection: {args.collection_name}")
    print(f"Models: {len(models)}")
    manager.print_models(models)


def list_providers(args):
    """List all available providers."""
    manager = ModelManager()
    providers = manager.get_providers()

    print(f"\nAvailable providers ({len(providers)}):")
    for provider in providers:
        models = manager.get_models_by_provider(provider)
        print(f"  - {provider}: {len(models)} models")


def list_collections(args):
    """List all available collections."""
    manager = ModelManager()
    collections = manager.get_collections_list()

    print(f"\nAvailable collections ({len(collections)}):")
    for coll in collections:
        models = manager.get_collection(coll)
        print(f"  - {coll}: {len(models)} models")


def chat(args):
    """Send a quick chat message to a model."""
    manager = ModelManager()

    try:
        model = manager.get_model(args.model_id)
    except KeyError as e:
        print(f"Error: {e}")
        return

    print(f"Using: {model.model_id} ({model.provider})")
    print(f"Message: {args.message}\n")

    client = manager.create_client()

    try:
        if args.stream:
            print("Response: ", end="", flush=True)
            stream = client.chat.completions.create(
                model=model.model_name,
                messages=[{"role": "user", "content": args.message}],
                stream=True,
                max_tokens=args.max_tokens,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
        else:
            response = client.chat.completions.create(
                model=model.model_name,
                messages=[{"role": "user", "content": args.message}],
                max_tokens=args.max_tokens,
            )
            print(f"Response:\n{response.choices[0].message.content}\n")
            print(f"Tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI for exploring and testing OpenRouter models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--provider", help="Filter by provider (anthropic, openai, google, meta)"
    )
    list_parser.add_argument(
        "--vision", action="store_true", help="Only show vision-capable models"
    )
    list_parser.add_argument(
        "--functions", action="store_true", help="Only show function-calling models"
    )
    list_parser.add_argument(
        "--min-context", type=int, help="Minimum context window size"
    )
    list_parser.set_defaults(func=list_models)

    # Show model command
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("model_id", help="Model ID to show")
    show_parser.set_defaults(func=show_model)

    # Collection command
    coll_parser = subparsers.add_parser("collection", help="Show collection")
    coll_parser.add_argument("collection_name", help="Collection name")
    coll_parser.set_defaults(func=show_collection)

    # Providers command
    prov_parser = subparsers.add_parser("providers", help="List all providers")
    prov_parser.set_defaults(func=list_providers)

    # Collections command
    colls_parser = subparsers.add_parser("collections", help="List all collections")
    colls_parser.set_defaults(func=list_collections)

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a message to a model")
    chat_parser.add_argument("model_id", help="Model ID to use")
    chat_parser.add_argument("message", help="Message to send")
    chat_parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens to generate"
    )
    chat_parser.add_argument(
        "--stream", action="store_true", help="Stream the response"
    )
    chat_parser.set_defaults(func=chat)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
