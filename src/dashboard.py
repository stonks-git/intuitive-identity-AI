"""Dashboard — localhost web interface for agent introspection.

Real-time view of the agent's cognitive state via SSE + REST API.
Serves a single-page dark-themed dashboard at http://0.0.0.0:8080.

v2: Terminal-style consciousness monitor. Two-column layout:
  - Left (65%): Conscious Mind — full LLM context in + response out, streaming
  - Right top (35%): Attention Queue — full text of competing candidates
  - Right bottom (35%): Memory Search — semantic search over memory DB

Uses aiohttp — runs as another coroutine in asyncio.gather alongside
the cognitive loop, consolidation, peripherals. Dashboard crash does
not kill the agent.

Memory browser uses direct asyncpg pool.fetch() for read-only access —
no side effects on the cognitive system (no access counts, no mutation).
Memory search uses search_hybrid(mutate=False) for the same reason.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

logger = logging.getLogger("agent.dashboard")

DASHBOARD_PORT = 8080


# ── Shared state ──────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Shared state readable by both cognitive loop and dashboard."""

    # Set by main.py at startup
    config: Any = None
    layers: Any = None
    memory: Any = None

    # Set by cognitive_loop after initialization
    attention: Any = None
    gut: Any = None
    safety: Any = None
    outcome_tracker: Any = None
    bootstrap: Any = None

    # Shared with cognitive_loop (same list object)
    conversation: list = field(default_factory=list)
    exchange_count: int = 0
    escalation_stats: dict = field(default_factory=lambda: {"retries": 0, "retry_successes": 0, "escalations": 0})

    # SSE broadcast
    _sse_subscribers: list = field(default_factory=list)

    def publish_event(self, event: dict):
        """Broadcast event to all SSE subscribers. Fire-and-forget."""
        dead = []
        for q in self._sse_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._sse_subscribers.remove(q)

    def subscribe(self) -> asyncio.Queue:
        """Register a new SSE subscriber. Returns its event queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._sse_subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove an SSE subscriber on disconnect."""
        if q in self._sse_subscribers:
            self._sse_subscribers.remove(q)


# ── Route handlers ────────────────────────────────────────────────────


async def index_handler(request):
    """Serve the dashboard HTML page."""
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def sse_handler(request):
    """Server-Sent Events stream for real-time dashboard updates."""
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    state: AgentState = request.app["agent_state"]
    q = state.subscribe()

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                data = json.dumps(event, default=_json_default)
                await response.write(
                    f"event: {event.get('type', 'update')}\ndata: {data}\n\n".encode()
                )
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
            except ConnectionResetError:
                break
    finally:
        state.unsubscribe(q)

    return response


async def api_status(request):
    """JSON snapshot of current agent state (header bar data)."""
    state: AgentState = request.app["agent_state"]

    data = {
        "agent_id": None,
        "phase": None,
        "model_s1": None,
        "model_s2": None,
        "memory_count": 0,
        "conversation_length": len(state.conversation),
        "exchange_count": state.exchange_count,
        "attention_queue_size": 0,
        "bootstrap": None,
        "gut": None,
        "energy": None,
        "escalation": state.escalation_stats,
    }

    if state.layers:
        data["agent_id"] = state.layers.manifest.get("agent_id")
        data["phase"] = state.layers.manifest.get("phase")

    if state.config:
        data["model_s1"] = state.config.models.system1.model
        data["model_s2"] = state.config.models.system2.model if state.config.models.system2 else None

    if state.memory:
        try:
            data["memory_count"] = await state.memory.memory_count()
        except Exception:
            pass

    if state.attention:
        data["attention_queue_size"] = state.attention.queue_size

    if state.bootstrap:
        achieved, total = state.bootstrap.progress
        data["bootstrap"] = {"achieved": achieved, "total": total}

    if state.gut:
        data["gut"] = {
            "summary": state.gut.gut_summary(),
            "charge": round(state.gut.emotional_charge, 3),
            "alignment": round(state.gut.emotional_alignment, 3),
        }

    try:
        from .llm import energy_tracker
        data["energy"] = {
            "session_cost": round(energy_tracker.session_cost, 6),
            "total_calls": len(energy_tracker.entries),
        }
    except Exception:
        pass

    return web.json_response(data)


async def api_attention(request):
    """Attention queue state — full text of all candidates."""
    state: AgentState = request.app["agent_state"]
    if not state.attention:
        return web.json_response({"queue": [], "queue_size": 0})

    queue_items = []
    for cand in state.attention._queue:
        queue_items.append({
            "content": cand.content,
            "source_tag": cand.source_tag,
            "urgency": round(cand.urgency, 3),
            "salience": round(cand.salience, 3),
            "created_at": cand.created_at.isoformat() if cand.created_at else None,
        })

    return web.json_response({
        "queue": queue_items,
        "queue_size": state.attention.queue_size,
    })


async def api_memories_search(request):
    """Search memories using hybrid vector+text search (read-only, no mutation)."""
    state: AgentState = request.app["agent_state"]
    if not state.memory:
        return web.json_response({"results": [], "query": ""})

    query = request.query.get("q", "").strip()
    if not query:
        # No query: return latest 10
        rows = await state.memory.pool.fetch(
            """
            SELECT id, LEFT(content, 200) as content_preview, type,
                   importance, depth_weight_alpha, depth_weight_beta,
                   access_count, created_at
            FROM memories ORDER BY created_at DESC LIMIT 10
            """,
        )
        results = [_row_to_memory(r) for r in rows]
        return web.json_response({"results": results, "query": ""})

    try:
        hits = await state.memory.search_hybrid(query=query, top_k=15, mutate=False)
        results = []
        for h in hits:
            alpha = h.get("depth_weight_alpha", 1.0) or 1.0
            beta = h.get("depth_weight_beta", 4.0) or 4.0
            results.append({
                "id": h.get("id"),
                "content_preview": (h.get("content") or "")[:200],
                "type": h.get("type"),
                "importance": round(h.get("importance", 0) or 0, 3),
                "depth_center": round(alpha / (alpha + beta), 3),
                "access_count": h.get("access_count", 0) or 0,
                "created_at": h["created_at"].isoformat() if h.get("created_at") else None,
                "score": round(h.get("rrf_score", 0) or 0, 4),
            })
        return web.json_response({"results": results, "query": query})
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")
        return web.json_response({"results": [], "query": query, "error": str(e)})


async def api_memory_detail(request):
    """Single memory full detail."""
    state: AgentState = request.app["agent_state"]
    if not state.memory or not state.memory.pool:
        return web.json_response({"error": "no memory store"}, status=503)

    mem_id = request.match_info["id"]
    row = await state.memory.pool.fetchrow(
        """
        SELECT id, content, type, confidence, importance,
               depth_weight_alpha, depth_weight_beta,
               access_count, last_accessed, created_at, updated_at,
               source, source_tag, tags, immutable, compressed,
               evidence_count, metadata
        FROM memories WHERE id = $1
        """,
        mem_id,
    )
    if not row:
        return web.json_response({"error": "not found"}, status=404)

    alpha = row["depth_weight_alpha"] or 1.0
    beta = row["depth_weight_beta"] or 4.0
    data = {
        "id": row["id"],
        "content": row["content"],
        "type": row["type"],
        "confidence": round(row["confidence"] or 0, 3),
        "importance": round(row["importance"] or 0, 3),
        "depth_center": round(alpha / (alpha + beta), 3),
        "depth_alpha": round(alpha, 2),
        "depth_beta": round(beta, 2),
        "access_count": row["access_count"] or 0,
        "last_accessed": row["last_accessed"].isoformat() if row["last_accessed"] else None,
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "source": row["source"],
        "source_tag": row["source_tag"],
        "tags": row["tags"],
        "immutable": row["immutable"],
        "compressed": row["compressed"],
        "evidence_count": row["evidence_count"] or 0,
        "metadata": row["metadata"],
    }
    return web.json_response(data)


# ── Server lifecycle ──────────────────────────────────────────────────


async def run_dashboard(agent_state: AgentState, shutdown_event: asyncio.Event):
    """Run the dashboard web server as an asyncio task."""
    app = web.Application()
    app["agent_state"] = agent_state

    app.router.add_get("/", index_handler)
    app.router.add_get("/events", sse_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/attention", api_attention)
    app.router.add_get("/api/memories/search", api_memories_search)
    app.router.add_get("/api/memory/{id}", api_memory_detail)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", DASHBOARD_PORT)

    try:
        await site.start()
        logger.info(f"Dashboard started on http://0.0.0.0:{DASHBOARD_PORT}")
        await shutdown_event.wait()
    except Exception as e:
        logger.error(f"Dashboard error: {e}", exc_info=True)
    finally:
        await runner.cleanup()
        logger.info("Dashboard stopped.")


# ── Helpers ───────────────────────────────────────────────────────────


def _json_default(obj):
    """JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__float__"):
        return float(obj)
    return str(obj)


def _row_to_memory(r):
    """Convert a DB row to a memory dict for the API."""
    alpha = r["depth_weight_alpha"] or 1.0 if "depth_weight_alpha" in r.keys() else 1.0
    beta = r["depth_weight_beta"] or 4.0 if "depth_weight_beta" in r.keys() else 4.0
    return {
        "id": r["id"],
        "content_preview": r.get("content_preview") or (r.get("content") or "")[:200],
        "type": r["type"],
        "importance": round(r.get("importance", 0) or 0, 3),
        "depth_center": round(alpha / (alpha + beta), 3),
        "access_count": r.get("access_count", 0) or 0,
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
    }


# ── Inline HTML/CSS/JS ────────────────────────────────────────────────
# All user-facing data is escaped via textContent in the JS.
# The dashboard is localhost-only (Tailscale).

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Consciousness</title>
<style>
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --border: #30363d;
  --text: #c9d1d9;
  --dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --orange: #d29922;
  --red: #f85149;
  --purple: #bc8cff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  background: var(--bg);
  color: var(--text);
  font-size: 13px;
  line-height: 1.5;
  height: 100vh;
  overflow: hidden;
}

/* ── Header ── */
.hdr {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 4px 12px;
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 12px;
  height: 28px;
  flex-shrink: 0;
}
.hdr .title { color: var(--accent); font-weight: 600; font-size: 13px; }
.hdr .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 3px; vertical-align: middle; }
.hdr .dot.on { background: var(--green); }
.hdr .dot.off { background: var(--red); }
.hdr .stat { color: var(--dim); }
.hdr .stat b { color: var(--text); font-weight: 500; }

/* ── Layout ── */
.layout {
  display: flex;
  height: calc(100vh - 28px);
}
.left {
  flex: 65;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.right {
  flex: 35;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Panel chrome ── */
.ptitle {
  background: var(--bg2);
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--dim);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.pbody {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}

/* ── Conscious Mind ── */
.cycle {
  border-bottom: 1px solid var(--border);
  padding: 0;
}
.cycle-hdr {
  padding: 6px 10px;
  font-size: 11px;
  color: var(--dim);
  background: var(--bg2);
  cursor: default;
  display: flex;
  gap: 8px;
  align-items: center;
}
.cycle-hdr .src { font-weight: 600; }
.cycle-hdr .src.ext { color: var(--accent); }
.cycle-hdr .src.int { color: var(--purple); }
.cycle-section {
  padding: 4px 10px;
  border-top: 1px solid var(--bg3);
}
.cycle-label {
  font-size: 11px;
  color: var(--dim);
  font-weight: 600;
  cursor: pointer;
  user-select: none;
  padding: 3px 0;
}
.cycle-label:hover { color: var(--text); }
.cycle-content {
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
  padding: 2px 0 6px 0;
}
.collapsed .cycle-content { display: none; }
.cycle-section.response {
  border-left: 2px solid var(--green);
  padding-left: 8px;
}
.cycle-section.response.s2 {
  border-left-color: var(--orange);
}
.response .cycle-label { color: var(--green); }
.response.s2 .cycle-label { color: var(--orange); }

/* ── Attention Queue ── */
.attn-panel {
  flex: 35;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.attn-item {
  padding: 6px 10px;
  border-bottom: 1px solid var(--bg3);
  font-size: 12px;
}
.attn-item .attn-meta {
  font-size: 11px;
  color: var(--dim);
  margin-bottom: 2px;
}
.attn-item .attn-meta .src { font-weight: 600; }

/* ── Memory Search ── */
.mem-panel {
  flex: 65;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.search-bar {
  padding: 6px 10px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.search-bar input {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 4px 8px;
  font-family: inherit;
  font-size: 12px;
  border-radius: 3px;
  outline: none;
}
.search-bar input:focus { border-color: var(--accent); }
.mem-item {
  padding: 6px 10px;
  border-bottom: 1px solid var(--bg3);
  font-size: 12px;
  cursor: pointer;
}
.mem-item:hover { background: var(--bg2); }
.mem-meta {
  font-size: 11px;
  color: var(--dim);
  display: flex;
  gap: 8px;
}
.mem-meta .type { color: var(--purple); font-weight: 500; }
.mem-preview {
  margin-top: 2px;
  white-space: pre-wrap;
  word-break: break-word;
}
.mem-expanded {
  display: none;
  margin-top: 6px;
  padding: 6px 8px;
  background: var(--bg2);
  border-radius: 3px;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
}
.mem-item.open .mem-expanded { display: block; }
.mem-item.open .mem-preview { display: none; }
.depth-bar {
  display: inline-block; width: 30px; height: 5px;
  background: var(--bg3); border-radius: 2px; overflow: hidden; vertical-align: middle;
}
.depth-fill { height: 100%; border-radius: 2px; }
.empty-msg { color: var(--dim); padding: 10px; font-size: 12px; }
</style>
</head>
<body>

<div class="hdr">
  <span class="title"><span class="dot" id="dot"></span> Agent Consciousness</span>
  <span class="stat" id="h-gut"></span>
  <span class="stat" id="h-boot"></span>
  <span class="stat" id="h-cost"></span>
  <span class="stat" id="h-esc"></span>
  <span class="stat" id="h-queue"></span>
  <span class="stat" id="h-mem"></span>
  <span class="stat" id="h-model"></span>
</div>

<div class="layout">
  <div class="left">
    <div class="ptitle">Conscious Mind</div>
    <div class="pbody" id="mind"></div>
  </div>
  <div class="right">
    <div class="attn-panel">
      <div class="ptitle">Attention Queue</div>
      <div class="pbody" id="attn"></div>
    </div>
    <div class="mem-panel">
      <div class="ptitle">Memory</div>
      <div class="search-bar">
        <input type="text" id="mem-q" placeholder="search memories..." />
      </div>
      <div class="pbody" id="mem-results"></div>
    </div>
  </div>
</div>

<script>
// All dynamic content via textContent / DOM construction. No innerHTML with API data.

var mind = document.getElementById('mind');
var attn = document.getElementById('attn');
var memResults = document.getElementById('mem-results');
var memInput = document.getElementById('mem-q');
var dot = document.getElementById('dot');

// Track current cycle being assembled from SSE events
var currentCycle = null;
var cycleCount = 0;

// ── SSE ──
function connectSSE() {
  var es = new EventSource('/events');
  es.onopen = function() { dot.className = 'dot on'; };
  es.onerror = function() { dot.className = 'dot off'; };

  es.addEventListener('cycle_start', function(e) {
    var d = JSON.parse(e.data);
    startCycle(d);
    updateAttnFromCycle(d);
  });

  es.addEventListener('context_assembled', function(e) {
    var d = JSON.parse(e.data);
    addContext(d);
  });

  es.addEventListener('llm_response', function(e) {
    var d = JSON.parse(e.data);
    addResponse(d);
  });

  es.addEventListener('escalation', function(e) {
    var d = JSON.parse(e.data);
    addEscalation(d);
  });

  es.addEventListener('gate_flush', function(e) {
    var d = JSON.parse(e.data);
    addGateFlush(d);
  });
}

// ── Cycle construction ──
function startCycle(d) {
  cycleCount++;
  var div = document.createElement('div');
  div.className = 'cycle';
  div.id = 'cycle-' + cycleCount;

  // Header line
  var hdr = document.createElement('div');
  hdr.className = 'cycle-hdr';

  var ts = document.createElement('span');
  ts.textContent = d.ts ? new Date(d.ts).toLocaleTimeString() : '';
  hdr.appendChild(ts);

  var src = document.createElement('span');
  var srcTag = d.winner ? d.winner.source : (d.source || '?');
  src.className = 'src ' + (srcTag.indexOf('external') >= 0 ? 'ext' : 'int');
  src.textContent = srcTag;
  hdr.appendChild(src);

  if (d.winner) {
    var sal = document.createElement('span');
    sal.textContent = 'sal:' + d.winner.salience;
    hdr.appendChild(sal);
  }

  var qs = document.createElement('span');
  qs.textContent = 'q:' + (d.queue_size || 0);
  hdr.appendChild(qs);

  if (d.losers && d.losers.length > 0) {
    var lo = document.createElement('span');
    lo.textContent = '+' + d.losers.length + ' losers';
    hdr.appendChild(lo);
  }

  div.appendChild(hdr);

  // Input section — what won attention
  if (d.winner && d.winner.content) {
    var sec = makeSection('INPUT', d.winner.content, false);
    div.appendChild(sec);
  }

  mind.appendChild(div);
  currentCycle = div;
  autoScroll();
}

function addContext(d) {
  if (!currentCycle) return;

  // System prompt (collapsible, collapsed by default)
  if (d.system_prompt) {
    var sp = makeSection('SYSTEM PROMPT [' + (d.identity_tokens||0) + ' id tokens, shift:' + (d.context_shift||0) + ']', d.system_prompt, true);
    currentCycle.appendChild(sp);
  }

  // Conversation window
  if (d.conversation && d.conversation.length > 0) {
    var convText = '';
    d.conversation.forEach(function(m) {
      var tag = m.source_tag ? ' [' + m.source_tag + ']' : '';
      convText += m.role + tag + ': ' + m.content + '\n\n';
    });
    var cv = makeSection('CONVERSATION (' + d.conversation.length + ' msgs)', convText.trim(), true);
    currentCycle.appendChild(cv);
  }

  autoScroll();
}

function addResponse(d) {
  if (!currentCycle) return;

  var label = (d.escalated ? 'S2' : 'S1') + ' RESPONSE';
  if (d.model) label += ' [' + d.model + ']';
  if (d.confidence) label += ' conf:' + d.confidence;

  var sec = document.createElement('div');
  sec.className = 'cycle-section response' + (d.escalated ? ' s2' : '');

  var lbl = document.createElement('div');
  lbl.className = 'cycle-label';
  lbl.textContent = label;
  sec.appendChild(lbl);

  var ct = document.createElement('div');
  ct.className = 'cycle-content';
  ct.textContent = d.reply || '';
  sec.appendChild(ct);

  currentCycle.appendChild(sec);
  autoScroll();
}

function addEscalation(d) {
  if (!currentCycle) return;
  var sec = makeSection('ESCALATION triggers:[' + (d.triggers||[]).join(', ') + '] conf:' + d.confidence, '', false);
  sec.style.borderLeft = '2px solid var(--red)';
  sec.style.paddingLeft = '8px';
  currentCycle.appendChild(sec);
  autoScroll();
}

function addGateFlush(d) {
  // Add as a standalone entry, not part of a cycle
  var div = document.createElement('div');
  div.className = 'cycle';
  var hdr = document.createElement('div');
  hdr.className = 'cycle-hdr';
  var ts = document.createElement('span');
  ts.textContent = d.ts ? new Date(d.ts).toLocaleTimeString() : '';
  hdr.appendChild(ts);
  var info = document.createElement('span');
  info.style.color = 'var(--orange)';
  info.style.fontWeight = '600';
  info.textContent = 'GATE FLUSH: ' + d.persisted + ' persisted, ' + d.dropped + ' dropped';
  hdr.appendChild(info);
  div.appendChild(hdr);
  mind.appendChild(div);
  autoScroll();
}

function makeSection(label, content, collapsed) {
  var sec = document.createElement('div');
  sec.className = 'cycle-section' + (collapsed ? ' collapsed' : '');

  var lbl = document.createElement('div');
  lbl.className = 'cycle-label';
  lbl.textContent = (collapsed ? '\u25b6 ' : '\u25bc ') + label;
  lbl.addEventListener('click', function() {
    var isCollapsed = sec.classList.contains('collapsed');
    sec.classList.toggle('collapsed');
    lbl.textContent = (isCollapsed ? '\u25bc ' : '\u25b6 ') + label;
  });
  sec.appendChild(lbl);

  var ct = document.createElement('div');
  ct.className = 'cycle-content';
  ct.textContent = content;
  sec.appendChild(ct);

  return sec;
}

// ── Auto-scroll with scroll-lock detection ──
var userScrolled = false;

mind.addEventListener('scroll', function() {
  var atBottom = mind.scrollHeight - mind.scrollTop - mind.clientHeight < 50;
  userScrolled = !atBottom;
});

function autoScroll() {
  if (!userScrolled) {
    mind.scrollTop = mind.scrollHeight;
  }
}

// ── Attention queue ──
function updateAttnFromCycle(d) {
  attn.replaceChildren();

  // Show winner
  if (d.winner) {
    var item = makeAttnItem(d.winner.source, d.winner.salience, d.winner.content, true);
    attn.appendChild(item);
  }

  // Show losers
  if (d.losers) {
    d.losers.forEach(function(l) {
      var item = makeAttnItem(l.source, l.salience, l.content, false);
      attn.appendChild(item);
    });
  }

  if (!d.winner && (!d.losers || d.losers.length === 0)) {
    var empty = document.createElement('div');
    empty.className = 'empty-msg';
    empty.textContent = 'queue empty';
    attn.appendChild(empty);
  }
}

function makeAttnItem(source, salience, content, isWinner) {
  var div = document.createElement('div');
  div.className = 'attn-item';
  if (isWinner) div.style.borderLeft = '2px solid var(--green)';

  var meta = document.createElement('div');
  meta.className = 'attn-meta';

  var src = document.createElement('span');
  src.className = 'src';
  src.style.color = source.indexOf('external') >= 0 ? 'var(--accent)' : 'var(--purple)';
  src.textContent = source;
  meta.appendChild(src);

  var sal = document.createElement('span');
  sal.textContent = ' sal:' + salience;
  meta.appendChild(sal);

  if (isWinner) {
    var tag = document.createElement('span');
    tag.style.color = 'var(--green)';
    tag.textContent = ' \u2713 winner';
    meta.appendChild(tag);
  }

  div.appendChild(meta);

  var ct = document.createElement('div');
  ct.textContent = content || '';
  ct.style.whiteSpace = 'pre-wrap';
  ct.style.wordBreak = 'break-word';
  div.appendChild(ct);

  return div;
}

// Also poll attention every 5s for between-cycle updates
setInterval(function() {
  fetch('/api/attention').then(function(r) { return r.json(); }).then(function(d) {
    if (d.queue && d.queue.length > 0) {
      attn.replaceChildren();
      d.queue.forEach(function(c) {
        var item = makeAttnItem(c.source_tag, c.salience, c.content, false);
        attn.appendChild(item);
      });
    }
  }).catch(function(){});
}, 5000);

// ── Memory search ──
var searchTimeout = null;

memInput.addEventListener('input', function() {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(doSearch, 300);
});
memInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') { clearTimeout(searchTimeout); doSearch(); }
});

function doSearch() {
  var q = memInput.value.trim();
  var url = '/api/memories/search' + (q ? '?q=' + encodeURIComponent(q) : '');
  fetch(url).then(function(r) { return r.json(); }).then(function(d) {
    renderMemResults(d.results || []);
  }).catch(function(){});
}

function renderMemResults(results) {
  memResults.replaceChildren();
  if (results.length === 0) {
    var empty = document.createElement('div');
    empty.className = 'empty-msg';
    empty.textContent = 'no results';
    memResults.appendChild(empty);
    return;
  }
  results.forEach(function(m) {
    var div = document.createElement('div');
    div.className = 'mem-item';

    var meta = document.createElement('div');
    meta.className = 'mem-meta';

    var type = document.createElement('span');
    type.className = 'type';
    type.textContent = m.type || '-';
    meta.appendChild(type);

    var pct = Math.round((m.depth_center || 0) * 100);
    var bar = document.createElement('span');
    bar.className = 'depth-bar';
    var fill = document.createElement('span');
    fill.className = 'depth-fill';
    fill.style.width = pct + '%';
    fill.style.background = pct > 70 ? 'var(--green)' : pct > 40 ? 'var(--orange)' : 'var(--dim)';
    bar.appendChild(fill);
    meta.appendChild(bar);

    var imp = document.createElement('span');
    imp.textContent = 'imp:' + (m.importance || 0);
    meta.appendChild(imp);

    var acc = document.createElement('span');
    acc.textContent = 'acc:' + (m.access_count || 0);
    meta.appendChild(acc);

    if (m.score) {
      var sc = document.createElement('span');
      sc.textContent = 'score:' + m.score;
      meta.appendChild(sc);
    }

    if (m.created_at) {
      var dt = document.createElement('span');
      dt.textContent = m.created_at.split('T')[0];
      meta.appendChild(dt);
    }

    div.appendChild(meta);

    var preview = document.createElement('div');
    preview.className = 'mem-preview';
    preview.textContent = m.content_preview || '';
    div.appendChild(preview);

    // Expanded view (loaded on click)
    var expanded = document.createElement('div');
    expanded.className = 'mem-expanded';
    div.appendChild(expanded);

    div.addEventListener('click', function() {
      if (div.classList.contains('open')) {
        div.classList.remove('open');
        return;
      }
      div.classList.add('open');
      if (expanded.textContent) return; // already loaded
      fetch('/api/memory/' + encodeURIComponent(m.id))
        .then(function(r) { return r.json(); })
        .then(function(full) {
          expanded.textContent = '';
          var fields = [
            'id: ' + full.id,
            'type: ' + full.type,
            'depth: ' + full.depth_center + ' (a=' + full.depth_alpha + ' b=' + full.depth_beta + ')',
            'importance: ' + full.importance + '  confidence: ' + full.confidence,
            'access: ' + full.access_count + '  evidence: ' + full.evidence_count,
            'source: ' + (full.source_tag || full.source || '-'),
            'created: ' + (full.created_at || '-'),
            'immutable: ' + full.immutable,
            '',
            full.content,
          ];
          if (full.compressed) fields.push('\n[compressed] ' + full.compressed);
          expanded.textContent = fields.join('\n');
        }).catch(function(){
          expanded.textContent = '[error loading]';
        });
    });

    memResults.appendChild(div);
  });
}

// ── Header status polling ──
function refreshHeader() {
  fetch('/api/status').then(function(r) { return r.json(); }).then(function(d) {
    setText('h-gut', d.gut ? 'gut:' + d.gut.charge : 'gut:-');
    setText('h-boot', d.bootstrap ? 'boot:' + d.bootstrap.achieved + '/' + d.bootstrap.total : 'boot:-');
    setText('h-cost', d.energy ? '$' + d.energy.session_cost.toFixed(4) : '$0');
    setText('h-esc', d.escalation ? 'esc:' + d.escalation.escalations : 'esc:0');
    setText('h-queue', 'q:' + (d.attention_queue_size || 0));
    setText('h-mem', 'mem:' + (d.memory_count || 0));
    setText('h-model', d.model_s1 ? d.model_s1.split('-').slice(-2).join('-') : '');
  }).catch(function(){});
}

function setText(id, val) {
  var el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Init ──
connectSSE();
refreshHeader();
doSearch(); // load latest memories on startup
setInterval(refreshHeader, 5000);
</script>
</body>
</html>
"""
