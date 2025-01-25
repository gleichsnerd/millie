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
load_dotenv()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """
    Florp Game Management CLI

    This tool helps manage the Florp game system, including:
    - Docker container management for Milvus
    - Database initialization and seeding
    - Migration management
    - Application server control
    """
    pass



@cli.command()
@click.option("--force/--no-force", default=False, 
              help="Force update all embeddings even if content hasn't changed")
def update_embeddings(force):
    """Update embeddings for all rules in the database."""
    echo("Updating embeddings...")
    try:
        config = RAGConfig()
        rag = RAGManager(config)
        updated = rag.update_embeddings(force=force)
        if updated:
            echo(f"✅ Updated embeddings for {len(updated)} rules")
        else:
            echo("ℹ️ No embeddings needed updating")
    except Exception as e:
        echo(f"❌ Error updating embeddings: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--name", prompt="Migration name", help="Name of the migration")
def generate_migration(name):
    """Generate a new migration file."""
    echo(f"Generating migration '{name}'...")
    try:
        migration_manager = MigrationManager()
        migration_file = migration_manager.generate_migration(name)
        echo(f"✅ Migration generated: {migration_file}")
    except Exception as e:
        echo(f"❌ Error generating migration: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def run_migrations():
    """Run all pending migrations."""
    echo("Running migrations...")
    try:
        migration_manager = MigrationManager()
        applied = migration_manager.run_migrations()
        if applied:
            echo(f"✅ Applied {len(applied)} migrations successfully!")
        else:
            echo("ℹ️ No pending migrations.")
    except Exception as e:
        echo(f"❌ Error running migrations: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode with pdb")
def run(debug):
    """Start a new game of Florp."""
    try:
        from ..game.master import GameMaster
        from ..config.rag_config import RAGConfig

        echo("Starting Florp...")
        echo("Initializing Game Master...")

        config = RAGConfig()
        game_master = GameMaster(config)

        if debug:
            import pdb
            echo("🐛 Debug mode enabled. Type 'n' to step through code, 'c' to continue, or 'h' for help.")
            pdb.set_trace()

        # Start the game
        intro = game_master.start_game()
        echo("\n" + intro + "\n")

        # Game loop
        while True:
            try:
                # Get player input
                action = click.prompt("What would you like to do?")

                if action.lower() in ["quit", "exit", "q"]:
                    echo("\nThanks for playing Florp!")
                    break

                # Process the action
                if debug:
                    pdb.set_trace()
                result = game_master.process_action("player1", action)
                echo("\n" + result + "\n")

            except KeyboardInterrupt:
                echo("\nThanks for playing Florp!")
                break
            except Exception as e:
                echo(f"❌ Error processing action: {str(e)}", err=True)
                if debug:
                    pdb.post_mortem()
                continue

    except Exception as e:
        echo(f"❌ Error starting game: {str(e)}", err=True)
        if debug:
            pdb.post_mortem()
        sys.exit(1)


@cli.command()
def init():
    """Initialize database with fresh data (combines init-db --force, migrations, and seed-db)."""
    echo("🚀 Initializing Florp...")

    try:
        # Run init-db with force flag
        echo("\n📦 Initializing database...")
        ctx = click.get_current_context()
        ctx.invoke(init_db, force=True)

        # Run migrations
        echo("\n🔄 Running migrations...")
        ctx.invoke(run_migrations)

        # Seed database without embeddings first
        echo("\n🌱 Seeding database...")
        ctx.invoke(seed_db, drop=False, skip_embeddings=True)
        
        # Generate embeddings
        echo("\n🧠 Generating embeddings...")
        ctx.invoke(update_embeddings, force=True)

        echo("\n✨ Florp initialized successfully!")

    except Exception as e:
        echo(f"\n❌ Error during initialization: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--debug/--no-debug", default=False, help="Enable debug mode with pdb")
def dev(debug):
    """Development command: reinitialize everything and start the game (combines init and run)."""
    echo("🎮 Starting Florp in development mode...")

    try:
        # Run init command
        ctx = click.get_current_context()
        ctx.invoke(init)

        # Start the game
        echo("\n🎲 Starting game...")
        ctx.invoke(run, debug=debug)

    except Exception as e:
        echo(f"\n❌ Error in development mode: {str(e)}", err=True)
        if debug:
            import pdb
            pdb.post_mortem()
        sys.exit(1)


@cli.command()
def verify_env():
    """Verify environment configuration."""
    echo("Checking environment configuration...")

    # Check if .env file exists
    if not os.path.exists(".env"):
        echo("⚠️ No .env file found. Creating from template...")
        if os.path.exists(".env.template"):
            import shutil

            shutil.copy(".env.template", ".env")
            echo("✅ Created .env file from template")
            echo("⚠️ Please edit .env file with your actual configuration values")
        else:
            echo("❌ No .env.template file found", err=True)
            sys.exit(1)

    # Check required variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings",
        "MILVUS_HOST": "Milvus database host",
        "MILVUS_PORT": "Milvus database port",
    }

    missing = []
    for var, description in required_vars.items():
        value = get_env_var(var)
        if not value:
            missing.append(var)
            echo(f"❌ Missing {var} ({description})")
        else:
            masked_value = (
                value[:4] + "*" * (len(value) - 4) if var == "OPENAI_API_KEY" else value
            )
            echo(f"✅ {var}: {masked_value}")

    # Validate configuration
    try:
        config = RAGConfig()
        echo("\n✅ Configuration validation successful!")
    except Exception as e:
        echo(f"\n❌ Configuration validation failed: {str(e)}", err=True)
        if missing:
            echo("\n⚠️ Please set the following variables in your .env file:")
            for var in missing:
                echo(f"  - {var}: {required_vars[var]}")
        sys.exit(1)

    # Check Docker
    try:
        result = run_docker_command(["docker", "info"], check=False)
        if result.returncode == 0:
            echo("✅ Docker is running")
        else:
            echo("❌ Docker is not running", err=True)
    except Exception:
        echo("❌ Docker is not installed or not running", err=True)

    # Check Milvus container
    result = run_docker_command(
        [
            "docker",
            "ps",
            "--filter",
            "name=milvus-standalone",
            "--format",
            "{{.Status}}",
        ],
        check=False,
    )
    if result.stdout.strip():
        if "Up" in result.stdout:
            echo("✅ Milvus container is running")
        else:
            echo("⚠️ Milvus container exists but is not running")
    else:
        echo("⚠️ Milvus container is not created")


@cli.group()
def migrate():
    """Manage database migrations."""
    pass

@migrate.command()
@click.argument("name", required=False)
def create(name):
    """Create a new migration file."""
    try:
        from ..schema.migration_manager import MigrationManager
        from ..config.rag_config import RAGConfig

        config = RAGConfig()
        manager = MigrationManager(config)
        migration_file = manager.create_migration(name)
        if migration_file:
            echo(f"✅ Created migration file: {migration_file}")
    except Exception as e:
        echo(f"❌ Error creating migration: {str(e)}", err=True)
        sys.exit(1)

@migrate.command()
def run():
    """Run all pending migrations."""
    try:
        from ..schema.migration_manager import MigrationManager
        from ..config.rag_config import RAGConfig

        config = RAGConfig()
        manager = MigrationManager(config)
        manager.run_migrations()
        echo("✅ Successfully ran migrations")
    except Exception as e:
        echo(f"❌ Error running migrations: {str(e)}", err=True)
        sys.exit(1)

@migrate.command()
def rebuild_history():
    """Rebuild schema history from existing migrations."""
    try:
        from ..schema.migration_manager import MigrationManager
        from ..config.rag_config import RAGConfig

        config = RAGConfig()
        manager = MigrationManager(config)
        manager.rebuild_schema_history()
        echo("✅ Successfully rebuilt schema history")
    except Exception as e:
        echo(f"❌ Error rebuilding schema history: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def db():
    """
    Manage database operations.

    Commands for managing the Milvus database:
    - drop: Drop all collections
    """
    pass

@db.command()
def drop():
    """Drop all collections from the database."""
    echo("Dropping all collections...")
    
    session = MilvusSession(host=get_env_var("MILVUS_HOST"), port=get_env_var("MILVUS_PORT"))
    for model_class in session.MODEL_CLASSES:
        collection_name = model_class.collection_name()
        try:
            collection = Collection(collection_name)
            collection.drop()
            echo(f"✅ Dropped collection {collection_name}")
        except Exception as e:
            echo(f"⚠️ Error dropping {collection_name}: {e}")
    
    echo("✅ All collections dropped successfully!")


if __name__ == "__main__":
    cli()
