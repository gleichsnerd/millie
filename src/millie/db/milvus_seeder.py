"""Decorator for marking Milvus seeder functions."""
from typing import Callable, Dict
from functools import wraps

# Registry to store all seeder functions
_SEEDERS: Dict[str, Callable] = {}

def milvus_seeder(func: Callable) -> Callable:
    """Decorator to mark a function as a Milvus seeder.
    
    Example:
        @milvus_seeder
        def seed_rules():
            rule = RuleModel(
                id="1",
                type="rule",
                section="section",
                description="description"
            )
            # Save rule to Milvus...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Register the original function, not the wrapper
    _SEEDERS[func.__name__] = func
    return func  # Return the original function instead of the wrapper 