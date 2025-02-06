"""Millie package."""
from .cli import cli, add_millie_commands
from .orm import MilvusModel, MillieMigrationModel

__all__ = ['cli', 'add_millie_commands', 'MilvusModel', 'MillieMigrationModel']

if __name__ == "__main__":
    cli()