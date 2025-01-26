import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from millie.cli import cli, add_millie_commands
from millie.cli.attu.manager import attu
from millie.cli.util import run_docker_command
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
    """Mock the Docker command runner."""
    with patch('millie.cli.util.run_docker_command') as mock:
        def command_side_effects(*args, **kwargs):
            command = args[0]
            if "ps" in command and "milvus-attu" in command:
                return MagicMock(stdout="", stderr="", returncode=1)
            elif "inspect" in command:
                return MagicMock(stdout='{"State": {"Status": "exited"}}', stderr="", returncode=0)
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
def test_start_command_milvus_not_running(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when Milvus is not running."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout="", stderr="", returncode=1)
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 1
    assert "❌ Milvus is not running" in result.output

@click_skip_py310()
def test_start_command_container_exists(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when container exists."""
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Started Attu" in result.output

@click_skip_py310()
def test_start_command_container_stopped(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu when container is stopped."""
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Started Attu" in result.output

@click_skip_py310()
def test_start_command_new_container(test_cli, cli_runner, mock_docker_command):
    """Test starting Attu with a new container."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout="", stderr="", returncode=0)
    result = cli_runner.invoke(test_cli, ["attu", "start"])
    assert result.exit_code == 0
    assert "✅ Started Attu" in result.output

@click_skip_py310()
def test_stop_command_success(test_cli, cli_runner, mock_docker_command):
    """Test stopping Attu successfully."""
    result = cli_runner.invoke(test_cli, ["attu", "stop"])
    assert result.exit_code == 0
    assert "✅ Stopped Attu" in result.output

@click_skip_py310()
def test_status_command_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is running."""
    mock_docker_command.return_value = MagicMock(stdout="Up 2 hours")
    
    result = cli_runner.invoke(cli, ['attu', 'status'])
    assert result.exit_code == 0
    assert "✅ Attu container is running!" in result.output
    mock_sleep.assert_not_called()

@click_skip_py310()
def test_status_command_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is not running."""
    mock_docker_command.return_value = MagicMock(stdout="")
    
    result = cli_runner.invoke(cli, ['attu', 'status'])
    assert result.exit_code == 0
    assert "ℹ️ Attu container does not exist." in result.output
    mock_sleep.assert_not_called() 