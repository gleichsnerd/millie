#!/usr/bin/env python3
import click
import os
import sys
import subprocess
from pathlib import Path
import time
from pymilvus import Collection
from dotenv import load_dotenv

from ..util import create_millie_network, echo, run_docker_command
load_dotenv()

@click.group()
def milvus():
    """
    Manage the Milvus Docker container.

    Commands for managing the Milvus Docker container:
    - start: Start the Milvus container
    - stop: Stop the Milvus container
    - status: Check container status
    """
    pass

@milvus.command()
@click.option(
    "--standalone/--cluster",
    default=True,
    help="Run Milvus in standalone or cluster mode",
)
def start(standalone):
    """Start Milvus Docker container."""
    echo("Starting Milvus container...")

    # Check if Docker is running
    try:
        run_docker_command(["docker", "info"])
    except subprocess.CalledProcessError:
        echo("❌ Docker is not running. Please start Docker first.", err=True)
        sys.exit(1)

    # Check if container already exists
    result = run_docker_command(
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

    if result.stdout.strip():
        if "Up" in result.stdout:
            echo("✅ Milvus container is already running!")
            return
        else:
            # Start existing container
            echo("Starting existing Milvus container...")
            run_docker_command(["docker", "start", "milvus-standalone"])
            echo("✅ Milvus container started!")
            return

    # Create volumes
    volumes = ["milvus-etcd", "milvus-minio"]
    for volume in volumes:
        run_docker_command(["docker", "volume", "create", volume])

    create_millie_network()

    if standalone:
        echo("Starting Milvus in standalone mode...")
        run_docker_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "milvus-standalone",
                "--network",
                "millie",
                "-p",
                f"{os.getenv('MILVUS_PORT')}:{os.getenv('MILVUS_PORT')}",
                "-p",
                "9091:9091",
                "-v",
                "milvus-etcd:/etcd",
                "-v",
                "milvus-minio:/minio",
                "-e",
                "ETCD_USE_EMBED=true",
                "-e",
                "COMMON_STORAGETYPE=local",
                f"milvusdb/milvus:v{os.getenv('MILVUS_VERSION')}",
                "milvus",
                "run",
                "standalone",
            ]
        )
    else:
        echo("❌ Cluster mode not yet implemented", err=True)
        sys.exit(1)

    echo("Waiting for Milvus to start...")
    time.sleep(20)  # Give Milvus time to initialize
    echo("✅ Milvus container started!")


@milvus.command()
def stop():
    """Stop Milvus Docker container."""
    echo("Stopping Milvus container...")

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

    if not result.stdout.strip():
        echo("ℹ️ Milvus container is not running.")
        return

    run_docker_command(["docker", "stop", "milvus-standalone"])
    echo("✅ Milvus container stopped!")


@milvus.command()
def status():
    """Check Milvus Docker container status."""
    result = run_docker_command(
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

    if not result.stdout.strip():
        echo("ℹ️ Milvus container does not exist.")
    elif "Up" in result.stdout:
        echo("✅ Milvus container is running!")
    else:
        echo("ℹ️ Milvus container is stopped.")
