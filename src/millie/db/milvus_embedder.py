"""Decorator for marking Milvus embedding functions."""
from typing import Callable, Dict
from functools import wraps

# Registry to store all embedder functions
_EMBEDDERS: Dict[str, Callable] = {}

def milvus_embedder(func: Callable) -> Callable:
    """Decorator to mark a function as a Milvus embedder.
    
    Example:
        @milvus_embedder
        def embed_rules():
            # Generate embeddings for rules
            for rule in rules:
                rule.embedding = generate_embedding(rule.description)
                
    The decorated function should handle its own database operations
    and embedding generation. The EmbeddingManager will simply run
    all decorated functions in sequence.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Register the original function, not the wrapper
    _EMBEDDERS[func.__name__] = func
    return func 