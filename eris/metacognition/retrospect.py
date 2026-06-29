"""Retrospective metacognition — Eris looks back over her OWN past thoughts.

A superhuman-but-legible cycle: it uses recall/breadth no human has (every
reflection on a topic held in view at once) and renders the result so the user
can read, challenge, and correct it. Material in = her past `Thought` records
(the thought-stream); output = a synthesis stored back, linked to what it
reviewed, so looking-back itself joins the trajectory.

The hard constraint that makes it trustworthy: every past thought the synthesis
references must resolve to a REAL thought-stream id — never a plausible-sounding
reconstruction. (The failure this forbids: "reviewing" a retrospective that
didn't exist, citing a `16:25 introspection` found nowhere in the input.)

This is NOT ingestion (no web `is_useful` gate — there are no external passages)
and NOT RAPTOR-over-the-stream (the stream is discrete reflections, not one long
document). RAPTOR belongs on the library; retrospection re-reads her thoughts.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional
import json
import re
import time
import uuid

from eris.memory.thought_stream import ThoughtStream, link_and_store
from eris.reasoning.calibration import verify_grounding


@dataclass
class Retrospective:
    id: str
    topic: str
    timestamp: float
    reviewed_ids: list            # the thought ids this synthesis is built from
    movement: str                 # "how my thinking moved" — cites [t:id]s
    now_grounded: list            # claims she now holds as fact/inference (tiered)
    still_open: list              # bridges/speculations NOT grounded — needs YOUR judgment
    mind_changes: list            # [{from_id, to_claim, why}] — revisions on real prior ids
    regime: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---- scoping ---------------------------------------------------------------
def gather_retrospective_material(thought_stream: ThoughtStream, *,
                                  topic: Optional[str] = None,
                                  since: Optional[float] = None,
                                  limit: int = 20) -> List:
    """Pull HER OWN past thoughts as the material to reflect on. Bounded on
    purpose — mush comes from dumping the whole pile."""
    if topic:
        items = thought_stream.active_by_topic(topic, limit=limit)
    elif since is not None:
        items = thought_stream.since(since)
    else:
        raise ValueError("retrospection must be scoped (topic or window) — "
                         "never the whole stream")
    return items[-limit:]


# ---- the grounding contract ------------------------------------------------
def build_grounded_context(items) -> List[dict]:
    """Each past thought becomes a citable unit with its REAL id. The model may
    ONLY reference these ids; anything it 'remembers' outside this set is not
    retrievable and must not be cited."""
    return [{"id": t.id, "ts": t.timestamp, "regime": t.regime,
             "claims": t.claims, "text": t.text} for t in items]


_CITE_RE = re.compile(r"\[t:([0-9a-f]+)\]")


def unresolved_ids(reflection: str, allowed_ids) -> set:
    """Every [t:<id>] the model emitted that is NOT in the allowed set — a
    fabricated past-thought reference."""
    cited = set(_CITE_RE.findall(reflection or ""))
    return cited - set(allowed_ids)


def _strip_unsupported(text: str, bad_ids: set) -> str:
    """Drop any sentence that cites a fabricated id (last-resort backstop after a
    regeneration still references a thought that isn't there)."""
    if not bad_ids:
        return text
    keep = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        cited = set(_CITE_RE.findall(sentence))
        if cited & bad_ids:
            continue
        keep.append(sentence)
    return " ".join(keep).strip()


# ---- prompts ---------------------------------------------------------------
def _retro_prompt(topic: str, ctx: List[dict]) -> str:
    lines = []
    for c in ctx:
        claims = "; ".join(f"{cl.get('tier','?')}: {cl.get('text','')}"
                           for cl in (c.get("claims") or [])) or "(no tiered claims)"
        lines.append(f"[t:{c['id']}] (regime {c.get('regime','?')}) {c['text'][:600]}\n"
                     f"    claims: {claims}")
    body = "\n\n".join(lines)
    return (
        f"You are reviewing your OWN past reflections on '{topic}', provided below, "
        f"each with an id. Synthesize how your thinking on this topic has moved.\n\n"
        f"You may ONLY refer to past thoughts that appear in this set, and when you "
        f"refer to one you MUST cite its id like [t:9f3a]. If you find yourself about "
        f"to describe a past thought that is not in the set, stop — you do not have "
        f"it; do not invent it. Reflect as yourself, on what is actually here.\n\n"
        f"YOUR PAST THOUGHTS:\n\n{body}\n\n"
        f"Return ONLY a JSON object with these fields:\n"
        f'  "movement": a few sentences on how your thinking moved, citing [t:id]s;\n'
        f'  "now_grounded": list of {{"text","tier","source_id"}} you now hold as '
        f'fact/inference (source_id MUST be one of the ids above);\n'
        f'  "still_open": list of {{"text","tier"}} bridges/speculations you have '
        f'NOT grounded — where the user\'s judgment is needed;\n'
        f'  "mind_changes": list of {{"from_id","to_claim","why"}} where your view '
        f'moved (from_id MUST be one of the ids above).\n'
        f"Tiers are: fact, inference, bridge, speculation. Be honest — most new "
        f"connections are bridges or speculations until grounded.")


def _retry_clause(bad_ids: set) -> str:
    return ("\n\nIMPORTANT: your previous draft cited ids that do NOT exist in the "
            f"provided set: {', '.join(sorted(bad_ids))}. Those thoughts are not "
            "yours to cite — remove every reference to them and cite only ids that "
            "appear above.")


# ---- parsing ---------------------------------------------------------------
def _parse_json(draft: str) -> dict:
    """Lenient: pull the first {...} block and parse it; degrade gracefully."""
    if not draft:
        return {}
    m = re.search(r"\{.*\}", draft, re.DOTALL)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


# ---- the cycle -------------------------------------------------------------
def run_retrospective(thought_stream: ThoughtStream, topic: str,
                      generate: Callable[[str], str],
                      embed: Callable[[str], object], *,
                      regime: str = "", limit: int = 20) -> Optional[Retrospective]:
    """Retrieve her past thoughts on `topic`, synthesize their movement under the
    grounding contract, store the synthesis back (linked to what it reviewed),
    and log any mind-changes as supersedes events. Returns None if there is not
    yet enough to look back over.

    `generate(prompt) -> str` is the model call; `embed(text) -> vector`."""
    items = gather_retrospective_material(thought_stream, topic=topic, limit=limit)
    if len(items) < 2:
        return None                          # nothing to look back over yet
    allowed = [t.id for t in items]
    ctx = build_grounded_context(items)

    prompt = _retro_prompt(topic, ctx)
    draft = generate(prompt) or ""
    bad = unresolved_ids(draft, allowed)
    if bad:                                  # one regeneration with the miss called out
        draft = generate(prompt + _retry_clause(bad)) or draft
        bad = unresolved_ids(draft, allowed)

    parsed = _parse_json(draft)
    movement = str(parsed.get("movement") or draft).strip()
    if bad:                                  # still fabricating → strip, don't ship
        movement = _strip_unsupported(movement, bad)

    now_grounded = [c for c in (parsed.get("now_grounded") or []) if isinstance(c, dict)]
    still_open = [c for c in (parsed.get("still_open") or []) if isinstance(c, dict)]
    mind_changes = [c for c in (parsed.get("mind_changes") or []) if isinstance(c, dict)]

    # Grounding check (Part B.1): a 'fact/grounded' claim must (1) cite a reviewed id AND
    # (2) be SUBSTANTIVELY supported by that thought's text — quote-and-verify, not mere
    # id-resolution. A claim citing a live id whose text doesn't back it is demoted (or kept as
    # 'inference' if only implied); only an actually-supported claim ships as 'fact'. The reviewed
    # thoughts' own text are the sources; `generate` is the local judge.
    source_texts = {t.id: t.text for t in items}
    now_grounded = [verify_grounding(dict(c), allowed, source_texts=source_texts, model=generate)
                    for c in now_grounded]

    retro = Retrospective(
        id=uuid.uuid4().hex[:12], topic=topic, timestamp=time.time(),
        reviewed_ids=list(allowed), movement=movement,
        now_grounded=now_grounded, still_open=still_open,
        mind_changes=[mc for mc in mind_changes if mc.get("from_id") in allowed],
        regime=regime)

    # Store the synthesis back — it joins the trajectory, linked to the ids it
    # reviewed (so a later retrospective can review THIS one).
    link_and_store(thought_stream, topic=topic, regime=regime,
                   text=movement or f"Retrospective on {topic}.",
                   embedding=embed(movement) if movement else None,
                   claims=(now_grounded + still_open) or None,
                   drew_on=allowed, prior=allowed)

    # Mind-changes become visible supersede events on REAL reviewed ids.
    for mc in retro.mind_changes:
        to_claim = str(mc.get("to_claim") or "").strip()
        if not to_claim:
            continue
        link_and_store(thought_stream, topic=topic, regime=regime, text=to_claim,
                       embedding=embed(to_claim),
                       claims=[{"text": to_claim, "tier": "inference"}],
                       prior=[mc["from_id"]], supersedes=mc["from_id"])

    return retro
