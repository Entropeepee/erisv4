# Contractor Layer — Sovereignty-Routed LLM Gateway + Tiered Backends

One control plane for erisv4's **non-sensitive** LLM calls, built *behind* the existing
mediator (wrap, don't replace), with a hard **sovereignty boundary enforced in routing**.
IP-sensitive work runs only on the local model with no possible egress; non-sensitive work
routes across a cost-tiered pool (local → free cloud → frontier-synthesis → cheap-paid).

> **Status:** built and tested, **default-OFF**. Inert unless you set `ERIS_GATEWAY_BASE_URL`.
> The sovereign/IP-sensitive path never touches any of this regardless of configuration.
> The hive-synthesis escalation is additionally gated behind `ERIS_HIVE_SYNTH_CLOUD` (the
> A/B remains its gate). Nothing here changes default local-Ollama behavior.

---

## Why it exists

erisv4 already abstracts backends (`LLMBackend` ABC + `OllamaBackend`, `OpenAIBackend(base_url=…)`,
`AnthropicBackend`, `GeminiBackend`, `CustomBackend`) behind `LLMMediator.add_backend/generate/
race/ensemble`, and `config.py` uses an `ERIS_*_BASE_URL` convention for optional
OpenAI-compatible local services. This layer adds a **gateway** behind that abstraction plus a
**sovereignty router** — it does not rewrite the mediator or the Tier-4 `CONTINUE/SWITCH/ESCALATE`
router.

## The sovereignty boundary (defense in depth — all three required)

Every routed call carries a `sensitivity` tag: `sovereign` (IP-sensitive) or `open`.

1. **Routing (in-process, `eris/interface/sovereignty.py`).** A `sovereign` call may select
   **only** a direct local backend. `assert_backend_allowed()` **raises `SovereigntyError`** if a
   sovereign call is ever handed a non-local backend — it fails closed, never silently downgrades.
   An unknown tag or unrecognized backend coerces to the safe side (sovereign / non-local).
2. **Gateway policy (`deploy/litellm/config.yaml`).** The LiteLLM config defines **no route**
   from a sovereign model-group to a cloud provider; `local-mirror` points only at on-box
   Ollama/vLLM.
3. **OS egress (`eris/interface/egress_guard.py`).** The sovereign worker runs behind a
   Windows Firewall outbound-block on its venv `python.exe` and calls `assert_isolated()` at
   startup — it probes outbound TCP to an external IP and **exits if it succeeds**. Loopback is
   unaffected, so a local Ollama call still works.

## Components

| Module | Role |
|---|---|
| `eris/interface/sovereignty.py` | `Sensitivity` tag, `is_local_backend`, `assert_backend_allowed` (fail-closed), `select_sovereign_backend` |
| `eris/interface/egress_guard.py` | `assert_isolated()` / `isolation_status()` — OS egress self-check |
| `eris/interface/gateway.py` | `ContractorGateway` (cost tiers), `CachingBackend`, `ClaudeAgentSDKBackend` (Option A) |
| `eris/interface/contractor.py` | `ContractorRouter` — `(sensitivity, tier) → backend`, decision→tier map, cost log |
| `eris/interface/hermes.py` | `HermesContractor` — optional sandboxed research worker (loopback, non-IP) |

## Cost tiers (the `open` path)

`CONTINUE → local`, `SWITCH → free`, `ESCALATE → cheap` (pure mapping in `ContractorRouter`).

- **free** (`ERIS_TIER_FREE`, default `free-pool`) — bulk per-specialist reasoning. Gemini-free /
  Groq / OpenRouter-free behind the gateway.
- **cheap** (`ERIS_TIER_CHEAP`, default `cheap-paid`) — overflow when free is rate-limited.
- **synth** (`ERIS_TIER_SYNTH`, default `synth`) — frontier synthesis. **Option A**: Claude via
  `claude-agent-sdk` on the **subscription credit** (OAuth, *not* an API key). This backend is
  separate from the gateway by design — a LiteLLM Anthropic route would bill pay-as-you-go and
  bypass the credit.

### ⚠️ Key-bypass guard
If `ANTHROPIC_API_KEY` is set, the Agent-SDK path would bill pay-go and bypass your credit, so
`ClaudeAgentSDKBackend` reports **unavailable** and `generate()` **raises** rather than silently
spend money. Unset `ANTHROPIC_API_KEY` to use the credit (Option A), or deliberately wire Option B.

## Configuration (all default-OFF / inert)

| Env var | Default | Meaning |
|---|---|---|
| `ERIS_GATEWAY_BASE_URL` | `""` | LiteLLM endpoint, e.g. `http://localhost:4000/v1`. Unset ⇒ gateway off. |
| `ERIS_GATEWAY_API_KEY` | `sk-litellm-local` | The gateway's own virtual key. |
| `ERIS_TIER_FREE` / `ERIS_TIER_CHEAP` / `ERIS_TIER_SYNTH` | `free-pool` / `cheap-paid` / `synth` | Model-group names. |
| `ERIS_HIVE_SYNTH_CLOUD` | `0` | Route hive synthesis to the synth tier. OFF ⇒ synthesis stays local. |
| `ERIS_HERMES_BASE_URL` / `ERIS_HERMES_API_KEY` | `""` | Sandboxed Hermes contractor (loopback only). Both unset ⇒ off. |
| `ERIS_SOVEREIGN_REQUIRE_ISOLATION` | `1` | Sovereign worker exits if egress is reachable. `0` skips (dev only, logged loudly). |

## How the hive uses it

`hive_research(topic, …, sensitivity="open")`:
- per-specialist reasoning (high volume) → **free** tier when the gateway is on and the call is
  `open` (else local);
- synthesis/canonize → **synth** tier only when `ERIS_HIVE_SYNTH_CLOUD=1` and `open` (else local);
- `sensitivity="sovereign"` → **every** call stays on the direct local model, fail-closed; the
  gateway is never touched. The per-run `tier_calls` field in the summary reports the tally.

## Running the gateway

See `deploy/litellm/config.yaml` (pin the image digest; loopback only). Then:
```
set ERIS_GATEWAY_BASE_URL=http://localhost:4000/v1
set ERIS_GATEWAY_API_KEY=sk-litellm-local
# optional, after the A/B says the hive earns it:
set ERIS_HIVE_SYNTH_CLOUD=1
```

## Drift from the spec (confirmed against live code)

- **§8.2** said register the gateway as `CustomBackend(base_url=…)`. `CustomBackend` takes a
  `url` + crude string `payload_template` and isn't OpenAI-shaped; `OpenAIBackend` already speaks
  the `/chat/completions` format LiteLLM exposes and takes `base_url`. → gateway tiers are
  `OpenAIBackend` instances renamed `gateway-*` (so they read as non-local). Matches the existing
  `embed_base_url` convention.
- **§5.3** said "reuse the two-node pattern / egress_guard." No such code existed → built fresh
  (`egress_guard.py`).
- **Router tiers** (`CONTINUE/SWITCH/ESCALATE → _local/_single_cloud_expert/_deep_ensemble`)
  confirmed in `orchestrator.py`; the tier mapping is provided as a pure lookup so it can be
  adopted without rewriting the live turn loop.

## Tests

- `test_sovereignty.py` — fail-closed routing + egress guard (14)
- `test_gateway.py` — tiers, failover cascade, caching, key-bypass guard (6)
- `test_contractor_routing.py` — `(sensitivity,tier)` resolution, sovereign-never-cloud, cost log, Hermes (12)
- `test_hive_contractor_wiring.py` — open→free, sovereign stays local, synth flag gates synth (3)
