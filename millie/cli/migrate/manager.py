#!/usr/bin/env python3
import click
import os
import sys
import subprocess
from pathlib import Path
import time
from click import echo
from pymilvus import Collection
from dotenv import load_dotenv

from millie.db.migration_history import MigrationHistoryModel
from millie.db.session import MilvusSession
load_dotenv()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def migrate():
    """
    Create, run, and manage migrations
    """
    pass

def check_migration_table():
    session = MilvusSession(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
    return session.collection_exists(MigrationHistoryModel.collection_name())
        
def enforce_migration_table():
    if not check_migration_table():
        echo("❌ Migration history table not found. Run `millie migrate init` to create it.")
        sys.exit(1)

@migrate.command()
def init():
    """Initialize the migration history table."""
    session = MilvusSession(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
    if check_migration_table():
        echo("✅ Migration history table already exists.")
    else:
        echo("🤖Creating migration history table")
        session.init_collection(MigrationHistoryModel)
        # session.load_collection(MigrationHistoryModel)
        echo("✅ Migration history table created.")
    
    session.unload_collection(MigrationHistoryModel)

@migrate.group()
def schema_history():
    """Manage schema history JSON that is saved to the repository"""
    pass

@schema_history.command()
def rebuild():
    """Rebuild schema history from existing migrations."""
    try:
        from millie import MigrationManager
        from ..config.rag_config import RAGConfig

        config = RAGConfig()
        manager = MigrationManager(config)
        manager.rebuild_schema_history()
        echo("✅ Successfully rebuilt schema history")
    except Exception as e:
        echo(f"❌ Error rebuilding schema history: {str(e)}", err=True)
        sys.exit(1)

@migrate.command()
@click.argument("name", required=False)
@click.argument("--dry/--no-dry", required=False)
def create(name, dry = False):
    """Create a new migration file."""
    try:
        from millie import MigrationManager
        manager = MigrationManager()
        migration_file = manager.generate(name)
        # if migration_file:
        #     echo(f"✅ Created migration file: {migration_file}")
    except Exception as e:
        echo(f"❌ Error creating migration: {str(e)}", err=True)
        sys.exit(1)

# @migrate.command()
# def run():
#     """Run all pending migrations."""
#     try:
#         from ..schema.migration_manager import MigrationManager
#         from ..config.rag_config import RAGConfig

#         config = RAGConfig()
#         manager = MigrationManager(config)
#         manager.run_migrations()
#         echo("✅ Successfully ran migrations")
#     except Exception as e:
#         echo(f"❌ Error running migrations: {str(e)}", err=True)
#         sys.exit(1)


