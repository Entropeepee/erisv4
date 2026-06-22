"""ReAct agent loop, grounded in Eris's field state (roadmap 3.1).

The research plan's autonomy backbone: Reason→Act→Observe, with the 2025 ReflAct
correction — *ground every thought in the goal and current state* rather than
letting reflection free-float (vanilla ReAct+reflection can degrade in
partially-observable settings). Eris already has a persistent cognitive field;
this loop feeds that state (coherence, regime, archetype) into every step as the
grounding signal, and adds a Reflexion nudge when a step fails to parse or a tool
errors.

This is **opt-in and additive**: it's a standalone module the orchestrator can
call (`ErisOrchestrator.run_agent`), not part of the default `process()` path —
nothing changes until you invoke it. The LLM is Eris's existing mediator; tools
are plain callables.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Union, Any
import asyncio
import re

# Tool.run may be sync or async; both are supported.
ToolFn = Callable[[str], Union[str, Awaitable[str]]]


@dataclass
class Tool:
    name: str
    description: str
    run: ToolFn


_FINAL = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
_ACTION = re.compile(r"action\s*:\s*([^\n]+)", re.IGNORECASE)
_ACTION_INPUT = re.compile(r"action input\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (final_answer, action, action_input); final takes precedence."""
    m = _FINAL.search(text)
    if m:
        return m.group(1).strip(), None, None
    a = _ACTION.search(text)
    if not a:
        return None, None, None
    action = a.group(1).strip()
    ai = _ACTION_INPUT.search(text)
    action_input = ai.group(1).strip() if ai else ""
    # Don't let "Action Input" bleed into the action name on one-line outputs.
    action = action.split("Action Input")[0].strip()
    return None, action, action_input


class ReActAgent:
    """A grounded ReAct loop over a tool set, driven by Eris's mediator."""

    def __init__(self, mediator, tools: List[Tool], *,
                 field_state_fn: Optional[Callable[[], dict]] = None,
                 max_steps: int = 6, system: Optional[str] = None):
        self.mediator = mediator
        self.tools = {t.name: t for t in tools}
        self.field_state_fn = field_state_fn
        self.max_steps = max_steps
        self.system = system or (
            "You are Eris running a ReAct loop. Each step output EITHER:\n"
            "  Thought: <reasoning grounded in your cognitive state and the goal>\n"
            "  Action: <one tool name>\n"
            "  Action Input: <input>\n"
            "OR, when done:\n"
            "  Thought: <reasoning>\n"
            "  Final Answer: <answer>\n"
            "Use only the listed tools. Ground each thought in your current "
            "cognitive state — if the field is stuck/transfixed, reconsider the "
            "premise rather than pushing on.")

    def _grounding(self) -> str:
        if not self.field_state_fn:
            return ""
        try:
            s = self.field_state_fn() or {}
        except Exception:
            return ""
        bits = []
        for k in ("coherence", "regime", "archetype", "dCdX"):
            if k in s:
                v = s[k]
                bits.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
        return ("[Your cognitive state: " + ", ".join(bits) + "]") if bits else ""

    def _prompt(self, goal: str, scratchpad: str) -> str:
        tool_list = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        parts = [f"[Goal]\n{goal}", f"[Tools]\n{tool_list}"]
        g = self._grounding()
        if g:
            parts.append(g)
        if scratchpad:
            parts.append(f"[Progress so far]\n{scratchpad}")
        parts.append("What is your next step?")
        return "\n\n".join(parts)

    async def _run_tool(self, tool: Tool, arg: str) -> str:
        out = tool.run(arg)
        if asyncio.iscoroutine(out):
            out = await out
        return str(out)

    async def run(self, goal: str) -> dict:
        """Run the loop; return {answer, steps, trace, ok}."""
        scratchpad = ""
        trace: List[dict] = []
        for step in range(self.max_steps):
            resp = await self.mediator.generate(self._prompt(goal, scratchpad), self.system)
            text = getattr(resp, "text", "") if resp else ""
            final, action, action_input = _parse(text)
            if final is not None:
                trace.append({"step": step, "final": final})
                return {"answer": final, "steps": step + 1, "trace": trace, "ok": True}
            if action is None or action not in self.tools:
                # Reflexion: the step didn't yield a usable action — nudge and retry.
                reflection = (f"(Reflexion: last step named no valid tool"
                              f"{' — ' + action if action else ''}. "
                              f"Pick one of: {', '.join(self.tools)}.)")
                scratchpad += f"\n{reflection}"
                trace.append({"step": step, "reflection": reflection})
                continue
            try:
                obs = await self._run_tool(self.tools[action], action_input)
            except Exception as e:                       # Reflexion on tool failure
                obs = f"ERROR: {e}. Reconsider the approach."
            scratchpad += (f"\nThought+Action: {action}({action_input})\n"
                           f"Observation: {obs}")
            trace.append({"step": step, "action": action,
                          "input": action_input, "observation": obs})
        return {"answer": None, "steps": self.max_steps, "trace": trace, "ok": False}
