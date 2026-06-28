"""Optional Inspect AI (UK AISI) adapter.

The brief recommends Inspect as the spine because its Solver abstraction can wrap a multi-step
pipeline and a bare model as two arms scored by the same Scorer on the same Dataset. This module
converts our BenchItems into Inspect Samples and sketches the two solvers, so you can run inside
Inspect if you prefer its log viewer / sandboxing. The self-contained runner (run.py) does the
same job without the dependency and is the primary, tested path.

Everything here imports `inspect_ai` lazily, so importing this module never requires it. The
Inspect API evolves across versions — treat these as a starting point and adjust to your
installed `inspect_ai`."""
from typing import Callable, List

from eris.experiments.benchmarks.core import BenchItem, build_prompt


def to_samples(items: List[BenchItem]):
    """BenchItem -> list[inspect_ai.dataset.Sample]. Multiple-choice items carry their choices;
    the target is the gold answer/letter."""
    from inspect_ai.dataset import Sample
    out = []
    for it in items:
        out.append(Sample(
            id=it.id,
            input=build_prompt(it),
            target=it.answer or ("UNANSWERABLE" if it.unanswerable else ""),
            choices=it.choices or None,
            metadata=dict(it.meta or {}),
        ))
    return out


def bare_solver(model_answer: Callable[[str], str]):
    """Arm A as an Inspect solver: one generate() call. `model_answer` is your bare endpoint
    callable; in pure Inspect you would instead use its built-in `generate()` solver with
    --model openai/<your-ollama-model>."""
    from inspect_ai.solver import solver, TaskState, Generate

    @solver
    def _bare():
        async def solve(state: "TaskState", generate: "Generate"):
            state.output.completion = model_answer(state.input_text)
            return state
        return solve
    return _bare()


def eris_solver(eris_answer: Callable[[str], str]):
    """Arm B as an Inspect solver: the full Eris pipeline. `eris_answer` binds to your live Eris
    (ingest the provided source + answer); see arms.eris_pipeline_arm."""
    from inspect_ai.solver import solver, TaskState, Generate

    @solver
    def _eris():
        async def solve(state: "TaskState", generate: "Generate"):
            state.output.completion = eris_answer(state.input_text)
            return state
        return solve
    return _eris()
