import os
import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli import cli, add_millie_commands
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
def mock_session():
    """Create a mock MilvusSession."""
    with patch('millie.db.session.MilvusSession') as mock:
        mock.return_value = MagicMock()
        mock.return_value.connect.return_value = True
        yield mock

@click_skip_py310()
def test_check_command_success(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database connection successfully."""
    result = cli_runner.invoke(test_cli, ["db", "check"])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'millie_sandbox'" in result.output

@click_skip_py310()
def test_check_command_with_options(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database connection with custom options."""
    result = cli_runner.invoke(test_cli, ["db", "check", "--db-name", "test", "--host", "localhost", "--port", "19530"])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'test'" in result.output

@click_skip_py310()
def test_check_command_failure(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database connection with failure."""
    mock_session.return_value.connect.side_effect = Exception("Test error")
    result = cli_runner.invoke(test_cli, ["db", "check"])
    assert result.exit_code == 1
    assert "❌ Error connecting to database: Test error" in result.output

@click_skip_py310()
def test_check_command_milvus_not_running(test_cli, cli_runner, mock_docker_command, mock_session):
    """Test checking database when Milvus is not running."""
    mock_docker_command.side_effect = lambda *args, **kwargs: MagicMock(stdout="", stderr="", returncode=1)
    result = cli_runner.invoke(test_cli, ["db", "check"])
    assert result.exit_code == 1
    assert "❌ Milvus is not running" in result.output 
