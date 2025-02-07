"""Millie package."""
from .cli import cli, add_millie_commands
from .orm import MilvusModel, MillieMigrationModel, milvus_field
from .db import milvus_embedder, milvus_seeder, EmbeddingManager, SeedManager
from .db.session import MilvusSession
__all__ = [
    'MilvusSession',
    'cli', 
    'add_millie_commands', 
    'MilvusModel', 
    'MillieMigrationModel', 
    'milvus_field',
    'milvus_embedder',
    'milvus_seeder',
    'EmbeddingManager',
    'SeedManager'
]

if __name__ == "__main__":
    cli()