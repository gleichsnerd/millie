from dataclasses import field as dataclass_field
from typing import Any, Callable, Optional, TypeVar, overload, Union, Type

T = TypeVar('T')

@overload
def field(*, default: T) -> T: ...

@overload
def field(*, default_factory: Callable[[], T]) -> T: ...

def field(*, default=None, default_factory=None, **kwargs):
    """Create a field definition for MilvusModel classes.
    
    Args:
        default: Default value for the field
        default_factory: Callable that returns a default value
        **kwargs: Additional field configuration options
        
    Returns:
        Field definition object
    """
    if default_factory is not None:
        return dataclass_field(default_factory=default_factory, **kwargs)
    return dataclass_field(default=default, **kwargs) 