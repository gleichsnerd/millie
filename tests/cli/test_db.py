import os
import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli.router import cli, add_millie_commands
from millie.cli.db.manager import db
from millie.db.session import MilvusSession
from millie.cli.util import run_docker_command
from tests.util import click_skip_py310

@pytest.fixture
def test_cli():
    """Create a fresh CLI for each test."""
    test_cli = click.Group('cli')
    test_cli.add_command(db)
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
            if isinstance(command, list):
                if "ps" in command and "milvus-standalone" in command and "--format" in command:
                    # Return that the container is running
                    return MagicMock(stdout="Up 2 hours", stderr="", returncode=0)
                elif "inspect" in command:
                    return MagicMock(stdout='{"State": {"Status": "running"}}', stderr="", returncode=0)
                elif "logs" in command:
                    return MagicMock(stdout="Started successfully", stderr="", returncode=0)
                elif "rm" in command:
                    return MagicMock(stdout="", stderr="", returncode=0)
            return MagicMock(stdout="", stderr="", returncode=0)
        mock.side_effect = command_side_effects
        yield mock

@pytest.fixture
def mock_session():
    """Create a mock MilvusSession."""
    with patch('millie.db.session.MilvusSession') as mock:
        with patch('millie.cli.db.manager.MilvusSession', mock):
            yield mock

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set required environment variables."""
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_DB_NAME", "millie_test")
    monkeypatch.setenv("MILLIE_SCHEMA_DIR", "schema")

@pytest.fixture(autouse=True)
def mock_start_milvus():
    """Mock the start_milvus function."""
    with patch('millie.cli.milvus.manager.start') as mock:
        mock.return_value = None
        yield mock

@pytest.fixture(autouse=True)
def mock_docker_utils():
    """Mock Docker utility functions."""
    with patch('millie.cli.db.manager.run_docker_command') as mock:
        def command_side_effects(*args, **kwargs):
            command = args[0]
            if isinstance(command, list):
                if "ps" in command and "name=milvus-standalone" in command:
                    # Return that the container is running
                    return MagicMock(stdout="Up 2 hours", stderr="", returncode=0)
                elif "inspect" in command:
                    return MagicMock(stdout='{"State": {"Status": "running"}}', stderr="", returncode=0)
                elif "logs" in command:
                    return MagicMock(stdout="Started successfully", stderr="", returncode=0)
                elif "rm" in command:
                    return MagicMock(stdout="", stderr="", returncode=0)
            return MagicMock(stdout="", stderr="", returncode=0)
        mock.side_effect = command_side_effects
        yield mock

@click_skip_py310()
def test_check_command_success(test_cli, cli_runner, mock_docker_command, mock_session, monkeypatch):
    """Test checking database connection successfully."""
    # Create a mock instance that will be returned when MilvusSession is instantiated
    session_instance = MagicMock()
    mock_session.return_value = session_instance
    
    # Ensure the environment variable is set (this will override any defaults in the CLI)
    monkeypatch.setenv("MILVUS_DB_NAME", "millie_sandbox")
    
    result = cli_runner.invoke(test_cli, ["db", "check"])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'millie_" in result.output

@click_skip_py310()
def test_check_command_with_options(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database connection with custom options."""
    # Create a mock instance that will be returned when MilvusSession is instantiated
    session_instance = MagicMock()
    mock_session.return_value = session_instance
    
    result = cli_runner.invoke(test_cli, ["db", "check", "--db-name", "test", "--host", "localhost", "--port", "19530"])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'test'" in result.output

@click_skip_py310()
def test_check_command_failure(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database connection with failure."""
    # Mock the MilvusSession class itself to raise an exception when instantiated
    mock_session.side_effect = Exception("Test error")
    
    result = cli_runner.invoke(test_cli, ["db", "check"])
    assert result.exit_code == 1
    assert "❌ Error connecting to database: Test error" in result.output

@click_skip_py310()
def test_check_command_milvus_not_running(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database when Milvus is not running."""
    def mock_side_effect(*args, **kwargs):
        command = args[0]
        if isinstance(command, list):
            if "ps" in command and "milvus-standalone" in command:
                return MagicMock(stdout="", stderr="", returncode=1)
            elif "inspect" in command:
                return MagicMock(stdout="", stderr="Container not found", returncode=1)
            elif "logs" in command:
                return MagicMock(stdout="", stderr="Container not found", returncode=1)
            elif "rm" in command:
                return MagicMock(stdout="", stderr="", returncode=0)
        return MagicMock(stdout="", stderr="", returncode=1)
    
    mock_docker_command.side_effect = mock_side_effect
    
    # Also update the mock_docker_utils fixture
    with patch('millie.cli.db.manager.run_docker_command') as mock:
        mock.side_effect = mock_side_effect
        result = cli_runner.invoke(test_cli, ["db", "check"])
        assert result.exit_code == 1
        assert "❌ Milvus is not running" in result.output 
