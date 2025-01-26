import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli.attu.manager import attu
from millie.cli.util import run_docker_command

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_docker_command():
    """Mock the Docker command runner."""
    with patch('millie.cli.attu.manager.run_docker_command') as mock:
        yield mock

@pytest.fixture
def mock_sleep():
    """Mock time.sleep to prevent hanging."""
    with patch('time.sleep') as mock:
        yield mock

def test_start_command_milvus_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when Milvus is not running."""
    mock_docker_command.return_value = MagicMock(stdout="")
    
    result = cli_runner.invoke(attu, ['start'])
    assert result.exit_code == 1
    assert "❌ Milvus must be running before starting Attu" in result.output
    mock_sleep.assert_not_called()

def test_start_command_container_exists(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when container already exists."""
    # Mock Milvus running check
    def command_side_effects(*args, **kwargs):
        if "milvus-standalone" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        elif "attu" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(attu, ['start'])
    assert result.exit_code == 0
    assert "✅ Attu container is already running!" in result.output
    assert "🌐 Access Attu at http://localhost:8000" in result.output
    mock_sleep.assert_not_called()

def test_start_command_container_stopped(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when container exists but is stopped."""
    # Mock Milvus running check
    def command_side_effects(*args, **kwargs):
        if "milvus-standalone" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        elif "attu" in " ".join(args[0]):
            return MagicMock(stdout="Exited")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(attu, ['start'])
    assert result.exit_code == 0
    assert "Removing existing Attu container..." in result.output
    assert "✅ Attu container started!" in result.output
    mock_sleep.assert_not_called()

def test_start_command_new_container(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when creating new container."""
    # Mock Milvus running check
    def command_side_effects(*args, **kwargs):
        if "milvus-standalone" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        elif "attu" in " ".join(args[0]):
            return MagicMock(stdout="")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(attu, ['start'])
    assert result.exit_code == 0
    assert "Starting Attu container..." in result.output
    assert "✅ Attu container started!" in result.output
    mock_sleep.assert_not_called()

def test_stop_command_success(cli_runner, mock_docker_command, mock_sleep):
    """Test stop command when container exists."""
    # Mock container running check
    def command_side_effects(*args, **kwargs):
        if "docker ps" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(attu, ['stop'])
    assert result.exit_code == 0
    assert "Stopping Attu container..." in result.output
    assert "✅ Attu container stopped!" in result.output
    mock_sleep.assert_not_called()

def test_status_command_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is running."""
    mock_docker_command.return_value = MagicMock(stdout="Up 2 hours")
    
    result = cli_runner.invoke(attu, ['status'])
    assert result.exit_code == 0
    assert "✅ Attu container is running!" in result.output
    mock_sleep.assert_not_called()

def test_status_command_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is not running."""
    mock_docker_command.return_value = MagicMock(stdout="")
    
    result = cli_runner.invoke(attu, ['status'])
    assert result.exit_code == 0
    assert "ℹ️ Attu container does not exist." in result.output
    mock_sleep.assert_not_called() 