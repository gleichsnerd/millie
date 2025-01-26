from dataclasses import dataclass, fields, field, MISSING, Field
from typing import Dict, Any, List, get_type_hints
from typeguard import check_type
from .base_model import BaseModel

def MilvusModel(is_migration_collection=False):
    """Decorator to convert a class into a MilvusModel dataclass."""
    
    def decorator(cls):
        # Get parent classes - if cls inherits from another decorated class, use that
        bases = [BaseModel]
        for base in cls.__bases__:
            if base is not object:
                bases.insert(0, base)
        
        # Create new class with proper inheritance
        class ModelWithBase(*bases):
            pass
        
        # Start with base class annotations
        annotations = {}
        for base in reversed(bases):
            if hasattr(base, '__annotations__'):
                annotations.update(base.__annotations__)
        
        # Add annotations from the decorated class
        annotations.update(get_type_hints(cls))
        ModelWithBase.__annotations__ = annotations
            
        # Copy any class attributes/methods that aren't in the annotations
        for key, value in cls.__dict__.items():
            if not key.startswith('__'):
                if key in ModelWithBase.__annotations__:
                    # If it's a field with a default value, create a field with that default
                    if not isinstance(value, Field):
                        value = field(default=value)
                setattr(ModelWithBase, key, value)
                
        # Add default values for parent class fields if not already set
        for base in bases:
            if hasattr(base, '__dataclass_fields__'):
                for fname, f in base.__dataclass_fields__.items():
                    if fname not in ModelWithBase.__dict__:
                        if f.default is not MISSING:
                            setattr(ModelWithBase, fname, field(default=f.default))
                        elif f.default_factory is not MISSING:
                            setattr(ModelWithBase, fname, field(default_factory=f.default_factory))
        
        # Make it a dataclass with kw_only=True and type checking
        decorated = dataclass(kw_only=True)(ModelWithBase)
        
        # Add runtime type checking to __init__
        original_init = decorated.__init__
        def type_checking_init(self, **kwargs):
            # Check types before calling original init
            hints = get_type_hints(decorated)
            for name, value in kwargs.items():
                if name in hints:
                    try:
                        check_type(value, hints[name])
                    except TypeError as e:
                        raise TypeError(f"Argument '{name}': {str(e)}")
            original_init(self, **kwargs)
        decorated.__init__ = type_checking_init
        
        # Set migration flag if needed
        if is_migration_collection:
            decorated.is_migration_collection = True
            
        # Preserve the original class name
        decorated.__name__ = cls.__name__
        
        return decorated
    
    return decorator