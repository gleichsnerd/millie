import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
import subprocess

from millie.cli.db.manager import db
from millie.db.session import MilvusSession

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_session():
    """Create a mock MilvusSession."""
    with patch('millie.cli.db.manager.MilvusSession') as mock:
        session_instance = MagicMock()
        mock.return_value = session_instance
        yield mock

@pytest.fixture
def mock_docker_command():
    """Create a mock for Docker commands."""
    with patch('millie.cli.db.manager.run_docker_command') as mock:
        def command_side_effects(*args, **kwargs):
            cmd = " ".join(args[0])
            if "ps" in cmd and "milvus-standalone" in cmd:
                return MagicMock(stdout="Up 2 hours", stderr="")
            elif "inspect" in cmd:
                return MagicMock(stdout='{"State": {"Running": true}}', stderr="", returncode=0)
            elif "logs" in cmd:
                return MagicMock(stdout="Started successfully", stderr="", returncode=0)
            return MagicMock(stdout="", stderr="", returncode=0)
        
        mock.side_effect = command_side_effects
        yield mock

def test_check_command_success(cli_runner, mock_session, mock_docker_command):
    """Test check command when database connection succeeds."""
    result = cli_runner.invoke(db, ['check'])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'millie_sandbox'" in result.output

def test_check_command_with_options(cli_runner, mock_session, mock_docker_command):
    """Test check command with custom options."""
    result = cli_runner.invoke(db, [
        'check',
        '--port', '19531',
        '--db-name', 'test_db',
        '--host', 'test_host'
    ])
    assert result.exit_code == 0
    assert "✅ Successfully connected to database 'test_db'" in result.output
    mock_session.assert_called_once_with(
        port='19531',
        db_name='test_db',
        host='test_host'
    )

def test_check_command_failure(cli_runner, mock_session, mock_docker_command):
    """Test check command when database connection fails."""
    mock_session.side_effect = Exception("Connection failed")
    
    result = cli_runner.invoke(db, ['check'])
    assert result.exit_code == 1
    assert "❌ Error connecting to database: Connection failed" in result.output

def test_check_command_milvus_not_running(cli_runner, mock_session, mock_docker_command):
    """Test check command when Milvus is not running."""
    def command_side_effects(*args, **kwargs):
        cmd = " ".join(args[0])
        if "ps" in cmd and "milvus-standalone" in cmd:
            return MagicMock(stdout="", stderr="")
        elif "rm" in cmd:
            return MagicMock(stdout="", stderr="", returncode=0)
        elif "inspect" in cmd:
            return MagicMock(stdout="", stderr="", returncode=1)
        return MagicMock(stdout="", stderr="", returncode=0)
    
    mock_docker_command.side_effect = command_side_effects
    
    with patch('millie.cli.milvus.manager.start') as mock_start:
        mock_start.side_effect = Exception("Failed to start Milvus")
        result = cli_runner.invoke(db, ['check'])
        assert result.exit_code == 1
        assert "❌ Failed to start Milvus" in result.output 
