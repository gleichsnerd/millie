#!/usr/bin/env python3
import click
import os
import sys
from click import echo
from dotenv import load_dotenv

from millie.db.migration_history import MigrationHistoryModel
from millie.db.session import MilvusSession
from millie.db.schema_history import SchemaHistory
from millie.db.migration_manager import MigrationManager

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def migrate():
    """
    Create, run, and manage migrations
    """
    pass

def check_migration_table():
    session = MilvusSession(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
    return session.collection_exists(MigrationHistoryModel)
        
def enforce_migration_table():
    if not check_migration_table():
        echo("❌ Migration history table not found. Run `millie migrate init` to create it.")
        sys.exit(1)

@migrate.command()
def init():
    """Initialize the migration history table."""
    try:
        session = MilvusSession(host=os.getenv("MILVUS_HOST"), port=os.getenv("MILVUS_PORT"))
        if check_migration_table():
            echo("✅ Migration history table already exists.")
        else:
            echo("🤖Creating migration history table")
            session.init_collection(MigrationHistoryModel)
            # session.load_collection(MigrationHistoryModel)
            echo("✅ Migration history table created.")
        
        session.unload_collection(MigrationHistoryModel)
    except Exception as e:
        echo(f"❌ Error connecting to database: {str(e)}", err=True)
        sys.exit(1)

@migrate.group()
def schema_history():
    """Manage schema history JSON that is saved to the repository"""
    pass

@schema_history.command()
def rebuild():
    """Rebuild schema history from existing migrations."""
    try:
        echo("🤖 Rebuilding schema history from migrations...")
        
        # Get schema directory from env or default
        schema_dir = os.getenv("MILLIE_SCHEMA_DIR", "schema")
        history_dir = os.path.join(schema_dir, "history")
        migrations_dir = os.path.join(schema_dir, "migrations")
        
        # Create schema history manager
        history = SchemaHistory(history_dir, migrations_dir)
        
        # Find all models
        manager = MigrationManager()
        models = manager._find_all_models()
        
        # Rebuild each model's schema
        for model in models:
            echo(f"Rebuilding schema for {model.__name__}...")
            schema = history.build_model_schema_from_migrations(model)
            history.save_model_schema(schema)
            echo(f"✅ Schema rebuilt for {model.__name__}")
            
        echo("\n✅ Successfully rebuilt all schema history files")
    except Exception as e:
        echo(f"❌ Error rebuilding schema history: {str(e)}", err=True)
        sys.exit(1)

@migrate.command()
@click.argument("name", required=False)
def create(name: str):
    """Create a new migration based on schema changes."""
    try:
        if not name:
            name = click.prompt("Enter a name for the migration")
            
        echo("Checking for schema changes...")
        manager = MigrationManager()        
        changes = manager.detect_changes()
        
        if not changes:
            echo("No schema changes detected.")
            sys.exit(0)
        
        has_real_changes = False
        for model_name, model_changes in changes.items():
            if model_changes["added"] or model_changes["removed"] or model_changes["modified"]:
                has_real_changes = True
                echo(f"\nChanges detected in {model_name}:")
                if model_changes["added"]:
                    echo("\nAdded fields:")
                    for field in model_changes["added"]:
                        echo(f"  + {field.name} ({field.dtype})")
                if model_changes["removed"]:
                    echo("\nRemoved fields:")
                    for field in model_changes["removed"]:
                        echo(f"  - {field.name} ({field.dtype})")
                if model_changes["modified"]:
                    echo("\nModified fields:")
                    for old, new in model_changes["modified"]:
                        echo(f"  ~ {old.name}: {old.dtype} -> {new.dtype}")
        
        if not has_real_changes:
            echo("No schema changes detected.")
            sys.exit(0)
            
        migration_path = manager.generate_migration(name)
            
        if migration_path:
            echo(f"✅ Created migration {migration_path}")
        else:
            echo("No migration created.")
            
    except Exception as e:
        echo(f"❌ Error creating migration: {str(e)}", err=True)
        sys.exit(1)

@migrate.command()
def run():
    """Run all pending migrations."""
    try:
        enforce_migration_table()
        
        echo("Running pending migrations...")
        manager = MigrationManager()
        applied = manager.run_migrations()
        
        if not applied:
            echo("No pending migrations to run.")
        else:
            for migration in applied:
                echo(f"✅ Applied migration: {os.path.basename(migration)}")
            echo("\n✅ Successfully ran all migrations")
            
    except Exception as e:
        echo(f"❌ Error running migrations: {str(e)}", err=True)
        sys.exit(1)