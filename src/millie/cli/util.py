import os
import subprocess
import sys

import click

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

def run_docker_command(cmd, check=True):
    """Run a docker command and return the result."""
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        if check:
            echo(f"❌ Docker command failed: {e.stderr}", err=True)
            sys.exit(1)
        e.returncode = 1  # Override the returncode to be 1 instead of 2
        return e

def check_millie_network():
    """Run a docker command to check if the millie network exists."""
    result = run_docker_command(["docker", "network", "inspect","millie"], check=False)
    return result.returncode == 0

def create_millie_network():
    """Run a docker command to create the millie network."""
    if not check_millie_network():
        echo("Creating `millie` network...")
        run_docker_command(["docker", "network", "create", "millie"])
        echo("`millie` network created!")
    else:
        echo("`millie` network already exists!")
