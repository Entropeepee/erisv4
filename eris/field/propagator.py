"""
Spatial Logic Gate Propagation (SLGP) Worker
=============================================

Background thread that:
1. Receives GateProgram objects from the BLC compiler
2. Applies gate changes to the hex lattice
3. Runs pulse propagation steps after each program

Thread-safe: uses a queue for incoming programs and a lock for
lattice state access. Can be started/stopped/paused for integration
with the dreaming loop and the main conversation pipeline.

Usage:
    from eris.field.propagator import SLGPWorker

    worker = SLGPWorker(lattice=grid)
    worker.start()
    worker.submit(gate_program)     # Non-blocking
    worker.pause()                  # For dreaming loop
    worker.resume()
    worker.stop()                   # Clean shutdown
"""

from __future__ import annotations
import threading
import queue
import time
from typing import Optional

from eris.field.lattice import HexLogicGrid
from eris.field.compiler import GateProgram


class SLGPWorker:
    """Background worker for applying gate programs and running propagation.

    Parameters
    ----------
    lattice : HexLogicGrid
        The hex grid to operate on.
    propagation_steps : int
        How many pulse propagation steps to run after applying each program.
    poll_interval : float
        How often (seconds) to check the queue when idle.
    """

    def __init__(
        self,
        lattice: HexLogicGrid,
        propagation_steps: int = 5,
        poll_interval: float = 0.1,
    ):
        self.lattice = lattice
        self.propagation_steps = propagation_steps
        self.poll_interval = poll_interval

        self._queue: queue.Queue[Optional[GateProgram]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # Statistics
        self.programs_applied: int = 0
        self.total_instructions_applied: int = 0

    def start(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="SLGP-Worker",
            daemon=True,  # Dies when main thread exits
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread cleanly."""
        self._running = False
        self._pause_event.set()  # Unpause if paused
        self._queue.put(None)    # Sentinel to wake up the queue
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def pause(self) -> None:
        """Pause processing (thread stays alive but blocks)."""
        self._paused = True
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume processing after pause."""
        self._paused = False
        self._pause_event.set()

    def submit(self, program: GateProgram) -> None:
        """Submit a gate program for async application. Non-blocking."""
        self._queue.put(program)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def _run_loop(self) -> None:
        """Main worker loop — runs in background thread."""
        while self._running:
            # Respect pause
            self._pause_event.wait()

            try:
                program = self._queue.get(timeout=self.poll_interval)
            except queue.Empty:
                continue

            if program is None:  # Sentinel for shutdown
                break

            self._apply_program(program)

    def _apply_program(self, program: GateProgram) -> None:
        """Apply a gate program to the lattice (thread-safe)."""
        with self._lock:
            applied = 0
            for instr in program.instructions:
                cell = self.lattice.get_cell(instr.q, instr.r)
                if cell is not None and 0 <= instr.direction < 6:
                    cell.gates[instr.direction] = instr.gate_type
                    applied += 1

            # Inject energy at instruction sites (to trigger propagation)
            for instr in program.instructions:
                self.lattice.inject_energy(instr.q, instr.r, instr.weight * 0.5)

            # Run propagation
            self.lattice.propagate(steps=self.propagation_steps)

            self.programs_applied += 1
            self.total_instructions_applied += applied

    def apply_immediate(self, program: GateProgram) -> None:
        """Apply synchronously (blocking). Use when you need the result now."""
        self._apply_program(program)

    def get_lattice_locked(self):
        """Context manager for safe access to the lattice from main thread.

        Usage:
            with worker.get_lattice_locked() as lattice:
                hotspots = lattice.find_hotspots()
        """
        return self._lock
