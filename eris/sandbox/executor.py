"""
Sandbox Executor — Isolated Code Execution
============================================

Ported from Eve2's sandbox architecture. Two execution modes:

    SUBPROCESS: Fast, lightweight. Runs in a child Python process
                with restricted imports. Good for quick tests.

    DOCKER:     Full isolation via container. Network disabled,
                filesystem sandboxed. For untrusted or heavy code.
                Requires Docker to be installed.

The sandbox enables Eris Echo to:
    - Test hypotheses about its own architecture
    - Run FRACTAL PDE experiments during dreaming
    - Validate code improvements before applying them
    - Execute numerical experiments for the research organ

Usage:
    from eris.sandbox.executor import SandboxExecutor

    sandbox = SandboxExecutor()
    result = sandbox.execute("print(sum(range(100)))")
    print(result.stdout)   # "4950"
    print(result.status)   # "completed"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import subprocess
import tempfile
import time
import os


class ExecutionMode(Enum):
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class ExecutionStatus(Enum):
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Failed validation


@dataclass
class ExecutionResult:
    """Result of a sandbox execution."""
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_ms: float = 0.0
    blocked_reason: str = ""


class SandboxExecutor:
    """Execute Python code in isolation.

    Default mode: SUBPROCESS (no Docker dependency).
    """

    def __init__(self,
                 mode: Optional[ExecutionMode] = None,
                 timeout: int = 60,
                 workspace_dir: Optional[str] = None,
                 python_path: str = "python"):
        # Default subprocess (no Docker dependency, and it has Eris's own libs so
        # she can run simulations of her own architecture). Set
        # ERIS_SANDBOX_MODE=docker for full container isolation of untrusted code.
        if mode is None:
            env_mode = os.environ.get("ERIS_SANDBOX_MODE", "subprocess").lower()
            mode = ExecutionMode.DOCKER if env_mode == "docker" else ExecutionMode.SUBPROCESS
        self.mode = mode
        self.timeout = timeout
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="eris_sandbox_")
        self.python_path = python_path
        os.makedirs(self.workspace_dir, exist_ok=True)

        # Statistics
        self.total_executions: int = 0
        self.successful: int = 0
        self.failed: int = 0
        self.blocked: int = 0

    def execute(self, code: str, timeout: Optional[int] = None,
                validate: bool = True) -> ExecutionResult:
        """Execute Python code in the sandbox.

        Parameters
        ----------
        code : str
            Python source code to execute.
        timeout : int, optional
            Override default timeout (seconds).
        validate : bool
            If True, run safety validation before execution.

        Returns
        -------
        ExecutionResult with status, stdout, stderr, timing.
        """
        self.total_executions += 1
        timeout = timeout or self.timeout

        # Validation
        if validate:
            from eris.sandbox.validator import validate_code
            is_safe, message = validate_code(code)
            if not is_safe:
                self.blocked += 1
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    blocked_reason=message,
                )

        # Dispatch to execution mode
        if self.mode == ExecutionMode.SUBPROCESS:
            result = self._execute_subprocess(code, timeout)
        elif self.mode == ExecutionMode.DOCKER:
            result = self._execute_docker(code, timeout)
        else:
            result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=f"Unknown execution mode: {self.mode}",
            )

        if result.status == ExecutionStatus.COMPLETED:
            self.successful += 1
        else:
            self.failed += 1

        return result

    def _execute_subprocess(self, code: str, timeout: int) -> ExecutionResult:
        """Execute in a child Python process."""
        # Write code to temp file
        code_path = os.path.join(self.workspace_dir, "_sandbox_exec.py")
        with open(code_path, "w") as f:
            f.write(code)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [self.python_path, code_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.workspace_dir,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )
            duration = (time.time() - t0) * 1000

            status = (ExecutionStatus.COMPLETED
                      if proc.returncode == 0
                      else ExecutionStatus.ERROR)

            return ExecutionResult(
                status=status,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Execution timed out after {timeout}s",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr=str(e),
                duration_ms=(time.time() - t0) * 1000,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(code_path)
            except OSError:
                pass

    def _execute_docker(self, code: str, timeout: int) -> ExecutionResult:
        """Execute in a Docker container (full isolation)."""
        code_path = os.path.join(self.workspace_dir, "_sandbox_exec.py")
        with open(code_path, "w") as f:
            f.write(code)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network=none",              # No network
                    "--memory=2g",                  # 2GB RAM limit
                    "--cpus=2",                     # 2 CPU cores max
                    "-v", f"{self.workspace_dir}:/workspace:rw",
                    "-w", "/workspace",
                    "python:3.11-slim",
                    "python", "/workspace/_sandbox_exec.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = (time.time() - t0) * 1000

            status = (ExecutionStatus.COMPLETED
                      if proc.returncode == 0
                      else ExecutionStatus.ERROR)

            return ExecutionResult(
                status=status,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stderr=f"Docker execution timed out after {timeout}s",
            )
        except FileNotFoundError:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stderr="Docker not found. Install Docker or use SUBPROCESS mode.",
            )
        finally:
            try:
                os.unlink(code_path)
            except OSError:
                pass

    def quick_test(self, code: str) -> tuple:
        """Quick test: returns (success: bool, output: str)."""
        result = self.execute(code, timeout=10)
        return (result.status == ExecutionStatus.COMPLETED,
                result.stdout if result.status == ExecutionStatus.COMPLETED
                else result.stderr or result.blocked_reason)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "blocked": self.blocked,
        }
