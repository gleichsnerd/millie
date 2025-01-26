import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli.milvus.manager import milvus
from millie.cli.util import run_docker_command

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_docker_command():
    """Mock the Docker command runner."""
    with patch('millie.cli.milvus.manager.run_docker_command') as mock:
        yield mock

@pytest.fixture
def mock_sleep():
    """Mock time.sleep to prevent hanging."""
    with patch('time.sleep') as mock:
        yield mock

def test_start_command_docker_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when Docker is not running."""
    mock_docker_command.side_effect = subprocess.CalledProcessError(1, "docker info")
    
    result = cli_runner.invoke(milvus, ['start'])
    assert result.exit_code == 1
    assert "❌ Docker is not running" in result.output
    mock_sleep.assert_not_called()

def test_start_command_container_exists(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when container already exists."""
    # Mock Docker info success
    mock_docker_command.return_value = MagicMock(stdout="")
    
    # Mock container exists check
    def command_side_effects(*args, **kwargs):
        if "docker ps -a" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(milvus, ['start'])
    assert result.exit_code == 0
    assert "✅ Milvus container is already running!" in result.output
    mock_sleep.assert_not_called()

def test_start_command_container_stopped(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when container exists but is stopped."""
    # Mock Docker info success
    mock_docker_command.return_value = MagicMock(stdout="")
    
    # Mock container exists check
    def command_side_effects(*args, **kwargs):
        if "docker ps -a" in " ".join(args[0]):
            return MagicMock(stdout="Exited")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(milvus, ['start'])
    assert result.exit_code == 0
    assert "Starting existing Milvus container..." in result.output
    assert "✅ Milvus container started!" in result.output
    mock_sleep.assert_not_called()

def test_start_command_new_container(cli_runner, mock_docker_command, mock_sleep):
    """Test start command when creating new container."""
    # Mock Docker info success
    mock_docker_command.return_value = MagicMock(stdout="")
    
    # Mock container does not exist
    def command_side_effects(*args, **kwargs):
        if "docker ps -a" in " ".join(args[0]):
            return MagicMock(stdout="")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(milvus, ['start'])
    assert result.exit_code == 0
    assert "Starting Milvus in standalone mode..." in result.output
    assert "✅ Milvus container started!" in result.output
    mock_sleep.assert_called_once_with(20)

def test_stop_command_success(cli_runner, mock_docker_command, mock_sleep):
    """Test stop command when container exists."""
    # Mock container running check
    def command_side_effects(*args, **kwargs):
        if "docker ps" in " ".join(args[0]):
            return MagicMock(stdout="Up 2 hours")
        return MagicMock(stdout="")
    
    mock_docker_command.side_effect = command_side_effects
    
    result = cli_runner.invoke(milvus, ['stop'])
    assert result.exit_code == 0
    assert "Stopping Milvus container..." in result.output
    assert "✅ Milvus container stopped!" in result.output
    mock_sleep.assert_not_called()

def test_status_command_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is running."""
    mock_docker_command.return_value = MagicMock(stdout="Up 2 hours")
    
    result = cli_runner.invoke(milvus, ['status'])
    assert result.exit_code == 0
    assert "✅ Milvus container is running" in result.output
    mock_sleep.assert_not_called()

def test_status_command_not_running(cli_runner, mock_docker_command, mock_sleep):
    """Test status command when container is not running."""
    mock_docker_command.return_value = MagicMock(stdout="")
    
    result = cli_runner.invoke(milvus, ['status'])
    assert result.exit_code == 0
    assert "ℹ️ Milvus container does not exist" in result.output
    mock_sleep.assert_not_called() 