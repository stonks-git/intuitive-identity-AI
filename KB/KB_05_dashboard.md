# KB 05: Dashboard

## Overview

Localhost web dashboard for real-time agent introspection. Serves at `http://0.0.0.0:8080`, accessible via Tailscale at `http://norisor:8080`.

Built with `aiohttp` — runs as another coroutine in `asyncio.gather` alongside the cognitive loop, consolidation, and peripherals. Dashboard crash does not kill the agent.

## Architecture

```
AgentState (dataclass)
  created in main.py
  written by cognitive_loop (assigns attention, gut, safety, etc.)
  read by dashboard handlers
  conversation list is shared (same object reference)
  SSE broadcast via asyncio.Queue per subscriber

run_dashboard()
  aiohttp web.Application
  binds 0.0.0.0:8080
  waits on shutdown_event
  access_log disabled
```

## Data Sharing

`AgentState` is created in `main.py` and passed to both `cognitive_loop()` and `run_dashboard()`.

- Loop assigns internal objects after creation: `agent_state.attention = attention`, etc.
- Conversation list is shared by reference: `conversation = agent_state.conversation`
- Exchange count synced: `agent_state.exchange_count = exchange_count`
- All reads are safe (single event loop, no thread contention).

## SSE Events

The cognitive loop publishes events at 4 points via `agent_state.publish_event()`:

| Event | When | Data |
|-------|------|------|
| `cycle_start` | After winner selected | source, content preview, salience, queue_size |
| `llm_response` | After LLM response | reply preview, confidence, escalated |
| `escalation` | Before System 2 call | triggers, confidence |
| `gate_flush` | After periodic flush | persisted count, dropped count |

Each browser tab gets its own `asyncio.Queue(maxsize=200)`. Events are fire-and-forget: `QueueFull` silently drops the subscriber. SSE keepalive every 15s.

## API Routes

```
GET /                  -> HTML dashboard (inline, dark theme)
GET /events            -> SSE stream (real-time cognitive events)
GET /api/status        -> JSON agent state snapshot
GET /api/memories      -> JSON paginated memory list (?limit=20&offset=0)
GET /api/memory/{id}   -> JSON single memory detail
GET /api/attention     -> JSON attention queue contents
GET /api/gut           -> JSON gut feeling state + delta log
GET /api/conversation  -> JSON current conversation window
GET /api/energy        -> JSON energy tracker breakdown
```

## Memory Browser

Uses direct `asyncpg pool.fetch()` with SELECT queries. Does NOT go through `MemoryStore` methods to avoid side effects:
- No access count updates
- No retrieval mutation
- The agent doesn't "feel" you browsing its memories

## Frontend

Single inline HTML/CSS/JS string (`DASHBOARD_HTML`). Dark theme, vanilla JS with `EventSource`.

4 panels:
1. **Live Feed** - scrolling SSE events (attention wins, LLM responses, gate flushes, escalations)
2. **Agent Status** - refreshes every 5s (phase, model, memory count, bootstrap, gut, energy cost)
3. **Context Window** - refreshes every 3s (current conversation messages)
4. **Memory Store** - paginated table with click-to-expand modal (refreshes every 10s)

All dynamic content uses `textContent` and DOM construction (no `innerHTML` with API data) to prevent XSS.

## Modules

### `src/dashboard.py` (~900 lines)

- `AgentState` dataclass with SSE broadcast
- Route handlers for all API endpoints
- `run_dashboard()` coroutine
- Inline HTML/CSS/JS

### `src/loop.py` (modified)

- Signature: `cognitive_loop(..., agent_state=None)`
- Assigns objects to `agent_state` after creation
- Uses shared conversation: `agent_state.conversation if agent_state else []`
- Publishes 4 SSE events at key points
- All agent_state operations guarded by `if agent_state:` (no-op without dashboard)

### `src/main.py` (modified)

- Creates `AgentState(config=config, layers=layers, memory=memory)`
- Passes `agent_state` to `cognitive_loop()`
- Adds `run_dashboard(agent_state, shutdown_event)` to tasks

## Security

- Binds to 0.0.0.0 but only accessible via Tailscale (not exposed to public internet)
- Read-only: no mutation endpoints, no write operations
- No authentication (Tailscale provides network-level auth)
- XSS prevented: all dynamic content via textContent/DOM construction

## Port

| Service | Port |
|---------|------|
| Dashboard | 8080 (mapped in docker-compose.yml) |

## Resilience

- `run_dashboard()` catches exceptions internally
- Dashboard crash logs error and exits without calling `shutdown_event.set()`
- Agent continues operating via Telegram/stdin without dashboard
- `agent_state=None` default means loop works identically without dashboard
