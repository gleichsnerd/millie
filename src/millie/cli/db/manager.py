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

from millie.db.session import MilvusSession
from millie.cli.util import run_docker_command
from millie.db.seed_manager import SeedManager
load_dotenv()

@click.group()
def db():
    """
    Manage the Milvus database.

    Commands for managing the Milvus Docker container:
    - init: Start the Milvus container
    - stop: Stop the Milvus container
    - status: Check container status
    """
    pass

@db.command()
@click.option("--port", default=os.getenv("MILVUS_PORT", 19530), help="The port of the Milvus server")
@click.option("--db-name", default=os.getenv("MILVUS_DB_NAME", "millie_sandbox"), help="The name of the Milvus database")
@click.option("--host", default=os.getenv("MILVUS_HOST", "localhost"), help="The host of the Milvus server")
def check(port, db_name, host):
    """Check connection to the database and if it exists"""
    try:
        # First check the container
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

        if not result.stdout.strip() or "Up" not in result.stdout:
            echo("❌ Milvus is not running. Starting it now...")
            try:
                # Check if container exists but is not running
                exists_result = run_docker_command(
                    [
                        "docker",
                        "ps",
                        "-a",
                        "--filter",
                        "name=milvus-standalone",
                        "--format",
                        "{{.Status}}",
                    ],
                    check=False,
                )
                if exists_result.stdout.strip():
                    echo("Found existing container in bad state. Removing it...")
                    run_docker_command(["docker", "rm", "-f", "milvus-standalone"])

                from ..milvus.manager import start as start_milvus
                start_milvus()

            except Exception as e:
                echo(f"❌ Failed to start Milvus: {str(e)}", err=True)
                sys.exit(1)

        # Verify container is running and ports are mapped correctly
        echo("Verifying Milvus container status...")
        inspect_result = run_docker_command(
            ["docker", "inspect", "milvus-standalone"], check=False
        )
        if inspect_result.returncode != 0:
            echo("❌ Failed to inspect Milvus container", err=True)
            sys.exit(1)

        # Check container logs for readiness
        logs_result = run_docker_command(
            ["docker", "logs", "milvus-standalone"], check=False
        )
        if "error" in logs_result.stdout.lower() or "error" in logs_result.stderr.lower():
            echo("⚠️ Warning: Potential errors found in Milvus logs:")
            echo(logs_result.stderr)

        echo(f"Attempting to connect to Milvus at {host}:{port}...")
        # Convert port to string to match test expectations
        MilvusSession(host=host, port=str(port), db_name=db_name)
        echo(f"✅ Successfully connected to database '{db_name}'")
        sys.exit(0)
    except Exception as e:
        echo(f"❌ Error connecting to database: {str(e)}", err=True)
        sys.exit(1)

@db.command()
def drop():
    """Drop all collections inside the database"""
    session = MilvusSession(host=os.getenv('MILVUS_HOST', 'localhost'), port=os.getenv('MILVUS_PORT', '19530'))

    try:
        echo("🗑️ Dropping existing collections...")
        session.drop_all_collections()
        echo("✅ All collections dropped successfully!")
    except Exception as e:
        echo(f"❌ Unable to drop collections: {str(e)}", err=True)
        sys.exit(1)

@db.command()
@click.option("--skip-embeddings/--with-embeddings", default=False, 
              help="Skip embedding generation (useful for development when rules haven't changed)")
def seed(skip_embeddings):
    """Seed the database with data from decorated classes"""
    manager = SeedManager()
    results = manager.run_seeders()
    
    # if not results:
    #     click.echo("No seeders found.")
    #     return
        
    # for name, result in results.items():
    #     if isinstance(result, str) and result.startswith("Error:"):
    #         click.echo(f"❌ {name}: {result}")
    #     else:
    #         click.echo(f"✅ {name} completed successfully")
