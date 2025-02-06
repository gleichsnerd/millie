"""Decorators for Milvus models."""
from typing import Type, TypeVar, Callable
from .milvus_model import MilvusModel

T = TypeVar('T', bound=MilvusModel)

def MillieMigrationModel(cls: Type[T]) -> Type[T]:
    """Decorator to mark a model as a migration collection.
    
    This decorator sets the is_migration_collection flag to True.
    Migration collections are used to track schema changes in the database.
    
    Example:
        @MillieMigrationModel
        class MyMigrationModel(MilvusModel):
            name: str
            version: str
    
    Returns:
        A decorated class marked as a migration collection
    """
    # Set migration collection flag as a class variable
    setattr(cls, 'is_migration_collection', True)
    return cls