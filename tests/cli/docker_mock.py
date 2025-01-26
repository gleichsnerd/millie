import subprocess
from typing import Dict, Any, List, Union
from dataclasses import dataclass

from millie.cli.util import echo

@dataclass
class MockResponse:
    """Response configuration for a docker command."""
    returncode: int
    stdout: str = ""
    stderr: str = ""
    exception: Exception = None

class DockerCommandMock:
    """Mock for Docker command execution with configurable responses.
    
    Usage:
        mock = DockerCommandMock()
        mock.set_responses({
            # Normal response
            "docker ps -a": MockResponse(
                returncode=0,
                stdout="container1",
                stderr=""
            ),
            # Exception response
            "docker info": MockResponse(
                returncode=1,
                stderr="Docker is not running",
                exception=subprocess.CalledProcessError(1, "docker info")
            )
        })
    """
    def __init__(self):
        self.responses: Dict[str, MockResponse] = {}
        self._print_calls = False

    def output_calls(self):
        self._print_calls = True

    def set_responses(self, responses: Dict[str, Union[MockResponse, subprocess.CompletedProcess]]):
        """Set response overrides for specific docker commands.
        
        Args:
            responses: Dict mapping command strings to either MockResponse or CompletedProcess objects.
                     CompletedProcess objects will be converted to MockResponse objects.
        """
        converted_responses = {}
        for cmd, response in responses.items():
            if isinstance(response, subprocess.CompletedProcess):
                converted_responses[cmd] = MockResponse(
                    returncode=response.returncode,
                    stdout=response.stdout,
                    stderr=response.stderr
                )
            else:
                converted_responses[cmd] = response
        self.responses = converted_responses
        
    def __call__(self, command: Union[str, List[str]], *args, **kwargs) -> subprocess.CompletedProcess:
        """Handle docker command execution with configured responses."""
        cmd_str = " ".join(command) if isinstance(command, list) else command
        # Check for exact matches first
        if cmd_str in self.responses:
            response = self.responses[cmd_str]
        else:
            # Check for partial matches (e.g., "docker ps" matches "docker ps -a")
            for pattern, resp in self.responses.items():
                if all(part in cmd_str for part in pattern.split()):
                    response = resp
                    break
            else:
                # Default response for unmocked calls
                response = MockResponse(
                    returncode=3,
                    stderr=f"Unmocked call: {cmd_str}"
                )
        
        # If an exception is configured, raise it
        if response.exception is not None:
            if isinstance(response.exception, subprocess.CalledProcessError):
                # Set the stderr and cmd if not already set
                if not response.exception.stderr:
                    response.exception.stderr = response.stderr
                if not response.exception.cmd:
                    response.exception.cmd = command
            raise response.exception
            
        if self._print_calls:
            print(f"\n\n[DockerCommandMock] Command called: {cmd_str}\n\n")
        # Return a CompletedProcess
        return subprocess.CompletedProcess(
            args=command,
            returncode=response.returncode,
            stdout=response.stdout,
            stderr=response.stderr
        ) 