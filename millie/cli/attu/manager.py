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

from ..util import create_millie_network, run_docker_command
load_dotenv()

# Force terminal output
os.environ["FORCE_COLOR"] = "1"

# Store original echo function
_original_echo = click.echo


def echo(*args, **kwargs):
    """Wrapper for click.echo that always enables color."""
    kwargs["color"] = True
    _original_echo(*args, **kwargs)


# Replace click.echo with our version
click.echo = echo

@click.group()
def attu():
    """
    Manage the Attu Docker container.

    Commands for managing the Attu Docker container:
    - start: Start the Attu container
    - stop: Stop the Attu container
    - status: Check container status
    """
    pass

@attu.command()
def start():
    """Start the Attu visualization tool for Milvus."""
    # First check if Milvus is running
    milvus_result = run_docker_command(
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
    if not milvus_result.stdout.strip() or "Up" not in milvus_result.stdout:
        echo(
            "❌ Milvus must be running before starting Attu. Please run `millie milvus start` first.",
            err=True,
        )
        sys.exit(1)

    # Check if Attu container exists
    attu_result = run_docker_command(
        ["docker", "ps", "-a", "--filter", "name=attu", "--format", "{{.Status}}"],
        check=False,
    )

    if attu_result.stdout.strip():
        if "Up" in attu_result.stdout:
            echo("✅ Attu container is already running!")
            echo("🌐 Access Attu at http://localhost:8000")
            echo("ℹ️  Connect using:")
            echo("   Host: localhost")
            echo(f"   Port: {os.getenv('MILVUS_PORT')}")
            return
        else:
            # Remove existing container
            echo("Removing existing Attu container...")
            run_docker_command(["docker", "rm", "attu"])

    # Start new Attu container
    echo("Starting Attu container...")

    milvus_version = os.getenv('MILVUS_VERSION')
    attu_version = 'latest'
    if milvus_version.startswith('2.4'):
        attu_version = 'v2.4'
    elif milvus_version.startswith('2.3'):
        attu_version = 'v2.3'
    elif milvus_version.startswith('2.2'):
        attu_version = 'v2.2'
    elif milvus_version.startswith('2.1'):
        attu_version = 'v2.2.2'
    else:
        echo("❌ Unsupported Milvus version. Please set MILVUS_VERSION to 2.2.10, 2.3.5, or 2.4.0")
        sys.exit(1)

    create_millie_network()

    try:
        run_docker_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "attu",
                "--network",
                "millie",
                "-p",
                "8000:3000",
                "-e",
                f"MILVUS_URL=milvus-standalone:{os.getenv('MILVUS_PORT')}",
                f"zilliz/attu:{attu_version}",
            ]
        )

        echo("✅ Attu container started!")
        echo("🌐 Access Attu at http://localhost:8000")
        echo("ℹ️  Connect using:")
        echo(f"   Address: milvus-standalone:{os.getenv('MILVUS_PORT')}")

    except Exception as e:
        echo(f"❌ Failed to start Attu: {str(e)}", err=True)
        sys.exit(1)


@attu.command()
def stop():
    """Stop the Attu visualization tool."""
    result = run_docker_command(
        ["docker", "ps", "--filter", "name=attu", "--format", "{{.Status}}"],
        check=False,
    )

    if not result.stdout.strip():
        echo("ℹ️ Attu container is not running.")
        return

    echo("Stopping Attu container...")
    run_docker_command(["docker", "stop", "attu"])
    echo("✅ Attu container stopped!")


@attu.command()
def status():
    """Check Attu Docker container status."""
    result = run_docker_command(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=attu",
            "--format",
            "{{.Status}}",
        ],
        check=False,
    )

    if not result.stdout.strip():
        echo("ℹ️ Attu container does not exist.")
    elif "Up" in result.stdout:
        echo("✅ Attu container is running!")
    else:
        echo("ℹ️ Attu container is stopped.")
