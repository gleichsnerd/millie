import os
import pytest
import subprocess
from unittest.mock import MagicMock, patch
from click.testing import CliRunner
from typing import Dict, Any, List, Union

from millie.cli.router import cli, add_millie_commands
from millie.cli.util import run_docker_command
from tests.cli.docker_mock import DockerCommandMock, MockResponse

def click_skip_py310():
    """Skip tests on Python 3.10 due to Click incompatibility."""
    return pytest.mark.skipif(
        os.getenv("PYTHON_VERSION") == "3.10",
        reason="Click tests are not compatible with Python 3.10",
    )

@pytest.fixture
def test_cli():
    """Create a fresh CLI for each test."""
    cli_obj = cli
    add_millie_commands(cli_obj)
    return cli_obj

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_docker_command():
    """Mock the Docker command runner with configurable responses."""
    mock = DockerCommandMock()
    with patch("subprocess.run", side_effect=mock) as patched:
        patched.set_responses = mock.set_responses
        patched.output_calls = mock.output_calls
        yield patched

@pytest.fixture
def mock_sleep(monkeypatch):
    """Mock time.sleep to prevent delays during tests."""
    mock = MagicMock()
    monkeypatch.setattr("time.sleep", mock)
    return mock

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set required environment variables for tests."""
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_VERSION", "2.3.3")

@click_skip_py310()
def test_start_command_docker_not_running(test_cli, cli_runner, mock_docker_command):
    """Test start command when Docker is not running."""
    mock_docker_command.set_responses({
        "docker info": MockResponse(
            returncode=1,
            stderr="Docker is not running",
            exception=subprocess.CalledProcessError(1, "docker info", stderr="Docker is not running")
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 1
    assert "Docker is not running" in result.output

@click_skip_py310()
def test_start_command_container_exists(test_cli, cli_runner, mock_docker_command):
    """Test starting Milvus when container exists and is running."""
    mock_docker_command.set_responses({
        "docker info": subprocess.CompletedProcess(
            args=["docker", "info"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker ps -a --filter name=milvus-standalone --format {{.Status}}": subprocess.CompletedProcess(
            args=["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "✅ Milvus container is already running!" in result.output

@click_skip_py310()
def test_start_command_container_stopped(test_cli, cli_runner, mock_docker_command):
    """Test starting Milvus when container is stopped."""
    mock_docker_command.set_responses({
        "docker info": subprocess.CompletedProcess(
            args=["docker", "info"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker ps -a --filter name=milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="Exited",
            stderr=""
        ),
        "docker start milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "start", "milvus-standalone"],
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "Starting existing Milvus container..." in result.output
    assert "✅ Milvus container started!" in result.output

@click_skip_py310()
def test_start_command_new_container(test_cli, cli_runner, mock_docker_command, mock_sleep):
    
    """Test starting Milvus with a new container."""
    mock_docker_command.set_responses({
        "docker info": subprocess.CompletedProcess(
            args=["docker", "info"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker ps -a --filter name=milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker volume create milvus-etcd": subprocess.CompletedProcess(
            args=["docker", "volume", "create", "milvus-etcd"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker volume create milvus-minio": subprocess.CompletedProcess(
            args=["docker", "volume", "create", "milvus-minio"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker network create millie": subprocess.CompletedProcess(
            args=["docker", "network", "create", "millie"],
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker run": subprocess.CompletedProcess(
            args=["docker", "run"],
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "Starting Milvus in standalone mode..." in result.output
    assert "✅ Milvus container started!" in result.output

@click_skip_py310()
def test_stop_command_success(test_cli, cli_runner, mock_docker_command):
    """Test stopping Milvus successfully."""
    mock_docker_command.set_responses({
        "docker ps --filter name=milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "ps", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        ),
        "docker stop milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "stop", "milvus-standalone"],
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "stop"])
    assert result.exit_code == 0
    assert "✅ Milvus container stopped!" in result.output

@click_skip_py310()
def test_status_command_running(test_cli, cli_runner, mock_docker_command):
    """Test checking Milvus status when running."""
    mock_docker_command.set_responses({
        "docker ps -a --filter name=milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "status"])
    assert result.exit_code == 0
    assert "✅ Milvus container is running!" in result.output

@click_skip_py310()
def test_status_command_not_running(test_cli, cli_runner, mock_docker_command):
    """Test checking Milvus status when not running."""
    mock_docker_command.set_responses({
        "docker ps -a --filter name=milvus-standalone": subprocess.CompletedProcess(
            args=["docker", "ps", "-a", "--filter", "name=milvus-standalone", "--format", "{{.Status}}"],
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    
    result = cli_runner.invoke(test_cli, ["milvus", "status"])
    assert result.exit_code == 0
    assert "ℹ️ Milvus container does not exist." in result.output 