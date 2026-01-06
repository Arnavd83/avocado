"""SSH client for remote instance management.

Provides SSH connectivity, command execution, and file transfer.
"""

import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import paramiko

from .config import get_config, load_env


class SSHError(Exception):
    """SSH operation error."""
    pass


class SSHClient:
    """SSH client for Lambda instances."""

    def __init__(
        self,
        host: str,
        user: str = "ubuntu",
        key_path: str | None = None,
        port: int = 22,
    ):
        """Initialize SSH client.

        Args:
            host: Remote host IP or hostname.
            user: SSH username (default: ubuntu for Lambda instances).
            key_path: Path to private SSH key. If None, reads from SSH_PRIVATE_KEY_PATH env.
            port: SSH port (default: 22).
        """
        load_env()
        self.host = host
        self.user = user
        self.port = port

        # Resolve key path
        if key_path is None:
            key_path = os.environ.get("SSH_PRIVATE_KEY_PATH", "~/.ssh/id_rsa")
        self.key_path = Path(key_path).expanduser()

        if not self.key_path.exists():
            raise SSHError(f"SSH key not found: {self.key_path}")

        self._client: paramiko.SSHClient | None = None

    def _get_client(self) -> paramiko.SSHClient:
        """Get or create paramiko SSH client."""
        if self._client is None:
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return self._client

    def connect(self, timeout: int = 30) -> None:
        """Establish SSH connection.

        Args:
            timeout: Connection timeout in seconds.

        Raises:
            SSHError: On connection failure.
        """
        client = self._get_client()
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=str(self.key_path),
                timeout=timeout,
                look_for_keys=False,
                allow_agent=False,
            )
        except paramiko.AuthenticationException as e:
            raise SSHError(f"SSH authentication failed: {e}")
        except paramiko.SSHException as e:
            raise SSHError(f"SSH error: {e}")
        except socket.error as e:
            raise SSHError(f"Connection error: {e}")

    def close(self) -> None:
        """Close SSH connection."""
        if self._client:
            self._client.close()
            self._client = None

    def is_connected(self) -> bool:
        """Check if SSH connection is active."""
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    def wait_for_ready(
        self,
        timeout: int = 300,
        interval: int = 5,
        max_attempts: int = 20,
        callback: Callable[[int, str], None] | None = None,
    ) -> bool:
        """Wait for SSH to become available.

        Args:
            timeout: Maximum seconds to wait.
            interval: Seconds between connection attempts.
            max_attempts: Maximum number of connection attempts (default: 20).
            callback: Optional callback(attempt, status) for progress.

        Returns:
            True when SSH is ready.

        Raises:
            SSHError: On timeout or max attempts exceeded.
        """
        start = time.time()
        attempt = 0

        while True:
            attempt += 1
            elapsed = time.time() - start

            if attempt > max_attempts:
                raise SSHError(
                    f"SSH not ready after {attempt} attempts "
                    f"(elapsed: {elapsed:.1f}s, max: {max_attempts})"
                )

            if elapsed > timeout:
                raise SSHError(f"SSH not ready after {timeout}s ({attempt} attempts)")

            try:
                # Try to connect
                if callback:
                    callback(attempt, "connecting")

                self.connect(timeout=min(10, interval))

                # Verify connection works with a simple command
                exit_code, stdout, stderr = self.run("echo ready", timeout=10)
                if exit_code == 0 and "ready" in stdout:
                    if callback:
                        callback(attempt, "ready")
                    return True

                self.close()

            except SSHError:
                self.close()
                if callback:
                    callback(attempt, "waiting")
                time.sleep(interval)
            except Exception as e:
                self.close()
                if callback:
                    callback(attempt, f"error: {e}")
                time.sleep(interval)

    def run(
        self,
        command: str,
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute command on remote host.

        Args:
            command: Command to execute.
            timeout: Command timeout in seconds.
            env: Optional environment variables to set.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            SSHError: On execution failure.
        """
        if not self.is_connected():
            self.connect()

        client = self._get_client()

        # Prepend environment variables if provided
        if env:
            env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
            command = f"{env_prefix} {command}"

        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Use a thread to enforce timeout on recv_exit_status()
            # recv_exit_status() can block indefinitely, so we need to wrap it
            exit_code = None
            exception_holder = [None]
            
            def get_exit_status():
                nonlocal exit_code
                try:
                    exit_code = stdout.channel.recv_exit_status()
                except Exception as e:
                    exception_holder[0] = e
            
            status_thread = threading.Thread(target=get_exit_status, daemon=True)
            status_thread.start()
            status_thread.join(timeout=timeout)
            
            if status_thread.is_alive():
                # Thread is still running - command timed out
                # Close the channel to force termination
                try:
                    stdout.channel.close()
                except:
                    pass
                raise SSHError(f"Command timed out after {timeout}s: {command[:100]}")
            
            if exception_holder[0]:
                raise SSHError(f"Command execution failed: {exception_holder[0]}")
            
            if exit_code is None:
                raise SSHError("Failed to get command exit status")
            
            stdout_str = stdout.read().decode("utf-8")
            stderr_str = stderr.read().decode("utf-8")
            return exit_code, stdout_str, stderr_str
        except SSHError:
            raise
        except paramiko.SSHException as e:
            raise SSHError(f"Command execution failed: {e}")
        except Exception as e:
            raise SSHError(f"Unexpected error executing command: {e}")

    def run_streaming(
        self,
        command: str,
        timeout: int = 60,
        callback: Callable[[str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute command on remote host with streaming output.

        Args:
            command: Command to execute.
            timeout: Command timeout in seconds.
            callback: Optional callback for each line of output.
            env: Optional environment variables to set.

        Returns:
            Tuple of (exit_code, stdout, stderr).

        Raises:
            SSHError: On execution failure.
        """
        if not self.is_connected():
            self.connect()

        client = self._get_client()

        # Prepend environment variables if provided
        if env:
            env_prefix = " ".join(f"{k}={v}" for k, v in env.items())
            command = f"{env_prefix} {command}"

        try:
            stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
            
            # Stream output in real-time by reading from channel
            stdout_buffer = []
            stderr_buffer = []
            stdout_lines = []
            stderr_lines = []
            exit_code = None
            exception_holder = [None]
            channel = stdout.channel
            
            def read_output():
                """Read from channel and buffer lines."""
                try:
                    while True:
                        # Check if channel is still open and has data
                        if channel.closed:
                            break
                        
                        # Read from stdout
                        if channel.recv_ready():
                            data = channel.recv(4096)
                            if data:
                                text = data.decode("utf-8", errors="replace")
                                stdout_buffer.append(text)
                                
                                # Process complete lines
                                lines = "".join(stdout_buffer).splitlines(True)
                                stdout_buffer.clear()
                                
                                # Keep incomplete line in buffer
                                if lines and not lines[-1].endswith('\n'):
                                    stdout_buffer.append(lines.pop())
                                
                                for line in lines:
                                    stdout_lines.append(line)
                                    if callback:
                                        stripped = line.rstrip()
                                        if stripped:
                                            callback(stripped)
                        
                        # Read from stderr
                        if channel.recv_stderr_ready():
                            data = channel.recv_stderr(4096)
                            if data:
                                text = data.decode("utf-8", errors="replace")
                                stderr_buffer.append(text)
                                
                                # Process complete lines
                                lines = "".join(stderr_buffer).splitlines(True)
                                stderr_buffer.clear()
                                
                                # Keep incomplete line in buffer
                                if lines and not lines[-1].endswith('\n'):
                                    stderr_buffer.append(lines.pop())
                                
                                for line in lines:
                                    stderr_lines.append(line)
                                    if callback:
                                        stripped = line.rstrip()
                                        if stripped:
                                            callback(stripped)
                        
                        # Check if command finished
                        if channel.exit_status_ready():
                            break
                        
                        # Small sleep to avoid busy waiting
                        time.sleep(0.1)
                        
                except Exception as e:
                    if not exception_holder[0]:
                        exception_holder[0] = e
            
            def get_exit_status():
                nonlocal exit_code
                try:
                    exit_code = channel.recv_exit_status()
                except Exception as e:
                    if not exception_holder[0]:
                        exception_holder[0] = e
            
            # Start threads
            read_thread = threading.Thread(target=read_output, daemon=True)
            status_thread = threading.Thread(target=get_exit_status, daemon=True)
            
            read_thread.start()
            status_thread.start()
            
            # Wait for command to complete
            status_thread.join(timeout=timeout)
            
            # Wait for read thread to finish
            read_thread.join(timeout=2)
            
            if status_thread.is_alive():
                try:
                    channel.close()
                except:
                    pass
                raise SSHError(f"Command timed out after {timeout}s: {command[:100]}")
            
            if exception_holder[0]:
                raise SSHError(f"Command execution failed: {exception_holder[0]}")
            
            if exit_code is None:
                raise SSHError("Failed to get command exit status")
            
            # Add any remaining buffered data
            if stdout_buffer:
                remaining = "".join(stdout_buffer)
                stdout_lines.append(remaining)
                if callback and remaining.strip():
                    callback(remaining.rstrip())
            
            if stderr_buffer:
                remaining = "".join(stderr_buffer)
                stderr_lines.append(remaining)
                if callback and remaining.strip():
                    callback(remaining.rstrip())
            
            # Read any remaining output
            try:
                remaining_stdout = stdout.read().decode("utf-8", errors="replace")
                remaining_stderr = stderr.read().decode("utf-8", errors="replace")
            except:
                remaining_stdout = ""
                remaining_stderr = ""
            
            stdout_str = "".join(stdout_lines) + remaining_stdout
            stderr_str = "".join(stderr_lines) + remaining_stderr
            
            return exit_code, stdout_str, stderr_str
        except SSHError:
            raise
        except paramiko.SSHException as e:
            raise SSHError(f"Command execution failed: {e}")
        except Exception as e:
            raise SSHError(f"Unexpected error executing command: {e}")

    def run_script(
        self,
        script_content: str,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a script on remote host.

        Args:
            script_content: Script content to execute.
            timeout: Script timeout in seconds.
            env: Optional environment variables.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        # Write script to temp file and execute
        script_path = "/tmp/inference_server_script.sh"

        # Escape script content for echo
        escaped = script_content.replace("'", "'\"'\"'")

        # Write script
        write_cmd = f"echo '{escaped}' > {script_path} && chmod +x {script_path}"
        exit_code, _, stderr = self.run(write_cmd, timeout=30)
        if exit_code != 0:
            raise SSHError(f"Failed to write script: {stderr}")

        # Execute script
        return self.run(f"bash {script_path}", timeout=timeout, env=env)

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
    ) -> None:
        """Upload a single file via SFTP.

        Args:
            local_path: Local file path.
            remote_path: Remote destination path.

        Raises:
            SSHError: On upload failure.
        """
        if not self.is_connected():
            self.connect()

        local_path = Path(local_path)
        if not local_path.exists():
            raise SSHError(f"Local file not found: {local_path}")

        client = self._get_client()
        try:
            sftp = client.open_sftp()
            sftp.put(str(local_path), remote_path)
            sftp.close()
        except Exception as e:
            raise SSHError(f"File upload failed: {e}")

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> None:
        """Download a single file via SFTP.

        Args:
            remote_path: Remote file path.
            local_path: Local destination path.

        Raises:
            SSHError: On download failure.
        """
        if not self.is_connected():
            self.connect()

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        client = self._get_client()
        try:
            sftp = client.open_sftp()
            sftp.get(remote_path, str(local_path))
            sftp.close()
        except Exception as e:
            raise SSHError(f"File download failed: {e}")

    def rsync(
        self,
        local_path: str | Path,
        remote_path: str,
        delete: bool = False,
        exclude: list[str] | None = None,
        dry_run: bool = False,
    ) -> tuple[int, str]:
        """Rsync local directory to remote.

        Uses system rsync command for efficient transfer.

        Args:
            local_path: Local directory path.
            remote_path: Remote destination path.
            delete: Delete files on remote that don't exist locally.
            exclude: Patterns to exclude.
            dry_run: Show what would be transferred without doing it.

        Returns:
            Tuple of (exit_code, output).

        Raises:
            SSHError: On rsync failure.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise SSHError(f"Local path not found: {local_path}")

        # Build rsync command
        cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",
            "-e", f"ssh -i {self.key_path} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        ]

        if delete:
            cmd.append("--delete")

        if dry_run:
            cmd.append("--dry-run")

        if exclude:
            for pattern in exclude:
                cmd.extend(["--exclude", pattern])

        # Ensure local path ends with / to copy contents
        local_str = str(local_path)
        if not local_str.endswith("/"):
            local_str += "/"

        cmd.append(local_str)
        cmd.append(f"{self.user}@{self.host}:{remote_path}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            raise SSHError("Rsync timed out")
        except Exception as e:
            raise SSHError(f"Rsync failed: {e}")

    def interactive_shell(self) -> None:
        """Open an interactive SSH shell.

        Uses system ssh command for proper terminal handling.
        """
        cmd = [
            "ssh",
            "-i", str(self.key_path),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{self.user}@{self.host}",
        ]

        # Replace current process with ssh
        os.execvp("ssh", cmd)

    def __enter__(self) -> "SSHClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def get_ssh_client_for_instance(instance_state) -> SSHClient:
    """Create SSH client for an instance from its state.

    Args:
        instance_state: InstanceState object.

    Returns:
        SSHClient configured for the instance.

    Raises:
        SSHError: If instance has no IP address.
    """
    ip = instance_state.public_ip
    if not ip:
        raise SSHError(f"Instance '{instance_state.name}' has no public IP")

    return SSHClient(host=ip, user="ubuntu")
