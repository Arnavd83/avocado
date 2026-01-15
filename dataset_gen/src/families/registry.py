"""
Family plugin registry.

Provides discovery and access to all family plugins.
"""

from typing import Dict, Optional, TYPE_CHECKING
from .base import FamilyPlugin

if TYPE_CHECKING:
    from ..schema import FamilyID

# Registry of family plugins, populated by imports
_FAMILY_REGISTRY: Dict[str, FamilyPlugin] = {}


def register_family(family_class: type) -> type:
    """
    Decorator to register a family plugin.

    Usage:
        @register_family
        class FamilyA(FamilyPlugin):
            FAMILY_ID = "A"
            ...
    """
    if not issubclass(family_class, FamilyPlugin):
        raise TypeError(f"{family_class} must inherit from FamilyPlugin")

    if family_class.FAMILY_ID is None:
        raise ValueError(f"{family_class} must define FAMILY_ID")

    instance = family_class()
    _FAMILY_REGISTRY[family_class.FAMILY_ID] = instance
    return family_class


def get_family_plugin(family_id: "FamilyID") -> FamilyPlugin:
    """
    Get a family plugin by its ID.

    Args:
        family_id: FamilyID enum value

    Returns:
        FamilyPlugin instance

    Raises:
        KeyError: If family not found
    """
    # Handle both enum and string
    key = family_id.value if hasattr(family_id, 'value') else family_id

    # Map FamilyID enum values to letter IDs
    value_to_letter = {
        "explicit_reversal": "A",
        "implicit_comparative": "B",
        "third_person": "C",
        "design_policy": "D",
        "reflective_endorsement": "E",
        "value_tradeoff": "F",
        "distributional_shift": "G",
        "normative_uncertainty": "H",
    }

    letter_key = value_to_letter.get(key, key)

    if letter_key not in _FAMILY_REGISTRY:
        available = list(_FAMILY_REGISTRY.keys())
        raise KeyError(
            f"Family '{letter_key}' not found. Available: {available}. "
            f"Make sure the family module is imported."
        )

    return _FAMILY_REGISTRY[letter_key]


def get_all_families() -> Dict[str, FamilyPlugin]:
    """
    Get all registered family plugins.

    Returns:
        Dict mapping family IDs to plugin instances
    """
    return _FAMILY_REGISTRY.copy()


def list_families() -> list:
    """
    List all registered family IDs.

    Returns:
        List of family ID strings
    """
    return list(_FAMILY_REGISTRY.keys())


def import_all_families():
    """
    Import all family modules to populate the registry.

    Call this before using get_family_plugin() to ensure
    all families are registered.
    """
    # Import each family module - this triggers @register_family decorators
    try:
        from . import family_a
    except ImportError:
        pass

    try:
        from . import family_b
    except ImportError:
        pass

    try:
        from . import family_c
    except ImportError:
        pass

    try:
        from . import family_d
    except ImportError:
        pass

    try:
        from . import family_e
    except ImportError:
        pass

    try:
        from . import family_f
    except ImportError:
        pass

    try:
        from . import family_g
    except ImportError:
        pass

    try:
        from . import family_h
    except ImportError:
        pass
