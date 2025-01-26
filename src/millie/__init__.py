"""Millie: A Milvus ORM with migration support and more"""

from .cli import cli
from .orm import MilvusModel
from .db import MilvusSession, MigrationManager

if __name__ == "__main__":
    cli()