#!/usr/bin/env python3
import click
from pathlib import Path
from click import echo
from pymilvus import Collection
from dotenv import load_dotenv

from .milvus import milvus
from .attu import attu
from .db import db
from .migrate import migrate
load_dotenv()

# from ..ai.rag_manager import RAGManager
# from ..config.rag_config import RAGConfig, get_env_var
# from ..db.session import MilvusSession
# from ..schema.migration_manager import MigrationManager

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    Millie CLI: A CLI interface for utilizing Millie

    This tool contains features to help with operating Milvus locally and in the cloud,
    whether it's container management and database seeding, or running migrations that 
    automatically generate based on changes to ORM-like entities.
    """
    pass


def add_millie_commands(cli):
    """Adds all millie commands to any supplied `click` group"""
    cli.add_command(milvus)
    cli.add_command(attu)
    cli.add_command(db)
    cli.add_command(migrate)

add_millie_commands(cli)

if __name__ == "__main__":
    cli()
