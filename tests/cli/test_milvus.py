import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli.router import cli, add_millie_commands
from millie.cli.milvus.manager import milvus
from millie.cli.util import run_docker_command
from tests.util import click_skip_py310

@pytest.fixture
def test_cli():
    """Create a fresh CLI for each test."""
    test_cli = click.Group('cli')
    test_cli.add_command(milvus)
    return test_cli

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_docker_command():
    """Mock the Docker command runner."""
    with patch('millie.cli.util.run_docker_command') as mock:
        def command_side_effects(*args, **kwargs):
            command = args[0]
            if "ps" in command and "milvus-standalone" in command:
                return MagicMock(stdout="Up 2 hours", stderr="", returncode=0)
            elif "inspect" in command:
                return MagicMock(stdout='{"State": {"Status": "running"}}', stderr="", returncode=0)
            elif "logs" in command:
                return MagicMock(stdout="Started successfully", stderr="", returncode=0)
            return MagicMock(stdout="", stderr="", returncode=0)
        mock.side_effect = command_side_effects
        yield mock

@pytest.fixture
def mock_sleep():
    """Mock time.sleep to prevent hanging."""
    with patch('time.sleep') as mock:
        yield mock

@click_skip_py310()
def test_start_command_docker_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when Docker is not running."""
    mock_docker_command.side_effect = subprocess.CalledProcessError(1, "docker info")
    
    result = cli_runner.invoke(cli, ['milvus', 'start'])
    assert result.exit_code == 1
    assert "❌ Docker is not running" in result.output
    mock_sleep.assert_not_called()

@click_skip_py310()
def test_start_command_container_exists(test_cli, cli_runner, mock_docker_command):
    """Test starting Milvus when container exists."""
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "✅ Milvus is already running" in result.output

@click_skip_py310()
def test_start_command_container_stopped(test_cli, cli_runner, mock_docker_command):
    """Test starting Milvus when container is stopped."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout='{"State": {"Status": "exited"}}', stderr="", returncode=0)
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "✅ Started Milvus" in result.output

@click_skip_py310()
def test_start_command_new_container(test_cli, cli_runner, mock_docker_command):
    """Test starting Milvus with a new container."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout="", stderr="", returncode=1)
    result = cli_runner.invoke(test_cli, ["milvus", "start"])
    assert result.exit_code == 0
    assert "✅ Started Milvus" in result.output

@click_skip_py310()
def test_stop_command_success(test_cli, cli_runner, mock_docker_command):
    """Test stopping Milvus successfully."""
    result = cli_runner.invoke(test_cli, ["milvus", "stop"])
    assert result.exit_code == 0
    assert "✅ Stopped Milvus" in result.output

@click_skip_py310()
def test_status_command_running(test_cli, cli_runner, mock_docker_command):
    """Test checking Milvus status when running."""
    result = cli_runner.invoke(test_cli, ["milvus", "status"])
    assert result.exit_code == 0
    assert "✅ Milvus is running" in result.output

@click_skip_py310()
def test_status_command_not_running(test_cli, cli_runner, mock_docker_command):
    """Test checking Milvus status when not running."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout="", stderr="", returncode=1)
    result = cli_runner.invoke(test_cli, ["milvus", "status"])
    assert result.exit_code == 1
    assert "❌ Milvus is not running" in result.output 