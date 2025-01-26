import subprocess
import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch

from millie.cli import cli, add_millie_commands
from millie.cli.attu.manager import attu
from tests.cli.docker_mock import DockerCommandMock, MockResponse
from tests.util import click_skip_py310

@pytest.fixture
def test_cli():
    """Create a fresh CLI for each test."""
    test_cli = click.Group('cli')
    test_cli.add_command(attu)
    return test_cli

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
    def mock_sleep(seconds):
        pass
    monkeypatch.setattr("time.sleep", mock_sleep)

@click_skip_py310()
def test_start_command_milvus_not_running(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when Milvus is not running."""
    mock_docker_command.set_responses({
        "docker ps --filter name=milvus-standalone": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 1
    assert "❌ Milvus must be running" in result.output

@click_skip_py310()
def test_start_command_container_exists(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when container exists."""
    mock_docker_command.set_responses({
        "docker ps --filter name=milvus-standalone --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        ),
        "docker ps -a --filter name=attu --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Exited",
            stderr=""
        ),
        "docker start attu": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Attu container started!" in result.output

@click_skip_py310()
def test_start_command_container_stopped(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when container is stopped."""
    mock_docker_command.set_responses({
        "docker ps --filter name=milvus-standalone --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        ),
        "docker ps -a --filter name=attu --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Exited",
            stderr=""
        ),
        "docker start attu": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Attu container started!" in result.output

@click_skip_py310()
def test_start_command_new_container(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu with a new container."""
    mock_docker_command.set_responses({
        "docker ps --filter name=milvus-standalone --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        ),
        "docker ps -a --filter name=attu --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        ),
        "docker run": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Attu container started!" in result.output

@click_skip_py310()
def test_stop_command_success(test_cli, cli_runner, mock_docker_command):
    """Test stopping Attu successfully."""
    mock_docker_command.set_responses({
        "docker ps --filter name=attu --format {{.Status}}": MockResponse(
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        ),
        "docker stop attu": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "stop"])
    assert result.exit_code == 0
    assert "✅ Attu container stopped!" in result.output

@click_skip_py310()
def test_status_command_running(test_cli, cli_runner, mock_docker_command):
    """Test status command when container is running."""
    mock_docker_command.set_responses({
        "docker ps -a --filter name=attu": MockResponse(
            returncode=0,
            stdout="Up 2 hours",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "status"])
    assert result.exit_code == 0
    assert "✅ Attu container is running!" in result.output

@click_skip_py310()
def test_status_command_not_running(test_cli, cli_runner, mock_docker_command):
    """Test status command when container is not running."""
    mock_docker_command.set_responses({
        "docker ps -a --filter name=attu": MockResponse(
            returncode=0,
            stdout="",
            stderr=""
        )
    })
    result = cli_runner.invoke(test_cli, ["attu", "status"])
    assert result.exit_code == 0
    assert "ℹ️ Attu container does not exist." in result.output 