# 10 — Peripheral Architecture: Body, Senses, and Voice

**Date:** 2026-02-11
**Status:** Design document for future implementation
**Depends on:** Cognitive loop (complete), Attention allocator (complete), Bootstrap readiness (complete)

---

## 1. Core Metaphor: Brain and Body

The cognitive loop is the brain. Everything else is a **peripheral** — a body part. The agent has one mind, one stream of consciousness, one context window. Peripherals are how that mind interacts with the world.

Each peripheral is a **driver** — a self-contained module that can be installed, configured, started, and stopped independently. Adding a new capability is like installing a driver for new hardware. The brain doesn't need to know the implementation details; it just knows what it can perceive and what it can do.

```
                    +------------------------------+
                    |       Cognitive Loop          |
                    |   (one mind, one stream)      |
                    |                               |
                    |   attention  <-- sensory[]    |
                    |   decisions  --> motor[]      |
                    +----------|-----|--------------+
                          sense|     |act
                    +----------|-----|--------------+
                    |     Capability Registry       |
                    |     (installed drivers)        |
                    +--+------+------+------+---+---+
                       |      |      |      |   |
                    +--+-+ +--+-+ +--+--+ +-+-+ +--+--+
                    | TG | | CLI| | Web | | FS| |Voice|
                    | bot| | tty| | eyes| |   | |     |
                    +----+ +----+ +-----+ +---+ +-----+
```

### 1.1 One Mind, Many Mouths

All peripherals share the same memory, identity, gut feeling, and consolidation engine. The agent is one entity talking through different windows. There are no per-channel context buffers, no parallel conversation histories. Everything flows through the ONE rolling context window.

If Alice messages on Telegram, then Bob messages, then a DMN thought fires, then Alice again — those all land in the same chronological stream, the same FIFO buffer, the same context the LLM sees. The agent is living one life, not running parallel sessions.

Separate per-channel context would create something like twins sharing a subconscious — multiple parallel conscious experiences with shared memory. That violates the single-consciousness design.

### 1.2 Return Address Routing

The brain has one output stream, but responses need to reach the right destination. Every input carries return address metadata:

```python
AttentionCandidate(
    content="Alice says: how are you?",
    source_tag="external_user",
    metadata={
        "capability": "telegram",
        "reply_to": {"chat_id": 123, "user": "alice"}
    }
)
```

When the cognitive loop produces a response to this input, it reads the return address and dispatches via the originating capability. If the agent also decides to browse something while thinking, that's a separate action — no routing confusion.

---

## 2. The Driver Interface

Every capability implements one contract:

```python
from typing import Protocol, AsyncIterator
from dataclasses import dataclass

@dataclass
class ActionSpec:
    """Declaration of something a capability can do."""
    name: str                    # "send_message", "navigate", "speak"
    params: dict[str, type]      # {"chat_id": int, "text": str}
    description: str             # Human-readable for system prompt injection
    reversible: bool = True      # Can this action be undone?
    requires_confirmation: bool = False  # Should the agent double-check?

@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    output: str                  # Text description of what happened
    sensory_data: dict | None    # Optional structured data (page content, etc.)
    error: str | None = None

class Capability(Protocol):
    name: str           # "telegram", "web_browser", "console", "voice"
    description: str    # For the agent's self-knowledge

    # --- Sensory: what can you perceive? (feeds attention queue) ---
    async def listen(self) -> AsyncIterator[AttentionCandidate]:
        """Yield inputs as they arrive. Runs continuously."""
        ...

    # --- Motor: what can you do? (brain calls these) ---
    def actions(self) -> list[ActionSpec]:
        """Declare available actions. Injected into system prompt."""
        ...

    async def execute(self, action: str, params: dict) -> ActionResult:
        """Execute a named action. Called by cognitive loop."""
        ...

    # --- Lifecycle ---
    async def start(self) -> None:
        """Initialize the capability (connect, authenticate, etc.)."""
        ...

    async def stop(self) -> None:
        """Clean shutdown."""
        ...

    # --- Introspection ---
    def status(self) -> dict:
        """Current state for cognitive state report."""
        ...
```

### 2.1 Capability Registry

The registry is the nervous system — it connects brain to body.

```python
class CapabilityRegistry:
    """Discovers, manages, and routes through installed capabilities."""

    capabilities: dict[str, Capability]  # name -> instance

    async def start_all(self) -> None:
        """Start all enabled capabilities."""

    async def stop_all(self) -> None:
        """Graceful shutdown of all capabilities."""

    def all_actions(self) -> list[ActionSpec]:
        """Collect actions from all capabilities. For system prompt injection."""

    async def unified_sensory_stream(self) -> AsyncIterator[AttentionCandidate]:
        """Merge listen() streams from all capabilities into one."""

    async def dispatch(self, capability: str, action: str, params: dict) -> ActionResult:
        """Route an action to the right capability."""

    def introspect(self) -> str:
        """What body parts do I have? For agent self-knowledge."""
```

### 2.2 Integration with Cognitive Loop

The cognitive loop currently reads from stdin and prints to stdout. The refactor:

1. **Input**: Instead of `asyncio.get_event_loop().run_in_executor(None, input)`, consume from `registry.unified_sensory_stream()`. All external messages, regardless of channel, arrive as AttentionCandidates and compete for attention alongside DMN, gut, and consolidation.

2. **Action dispatch**: When the LLM response includes tool calls (capability actions), the loop dispatches them through the registry. Results feed back into the LLM context as tool results.

3. **Response routing**: The loop reads `reply_to` metadata from the attention winner and dispatches the text response through the originating capability.

4. **System prompt**: Available actions from all capabilities are injected into the system prompt so the agent knows what it can do. This list is dynamic — if a capability starts or stops, the agent's awareness of its own body updates.

---

## 3. Planned Capabilities

### 3.1 CLI (extract from current loop.py)

**File:** `src/capabilities/cli.py`

The current stdin/stdout interaction extracted into a proper capability.

- **Sensory:** Yields lines from stdin as AttentionCandidates with `source_tag="external_user"`, `metadata={"capability": "cli"}`
- **Motor actions:**
  - `print(text)` — output to stdout
  - `print_status(text)` — formatted status output
- **Notes:** This is the bootstrap capability — always available, always works.

### 3.2 Telegram

**File:** `src/capabilities/telegram.py`
**Library:** aiogram (async-native, fits asyncio stack)

- **Sensory:** Incoming messages yield AttentionCandidates with `metadata={"capability": "telegram", "reply_to": {"chat_id": N, "user": "name", "username": "handle"}}`
- **Motor actions:**
  - `send_message(chat_id, text)` — reply to a chat
  - `send_photo(chat_id, image_path)` — send an image
  - `send_document(chat_id, file_path)` — send a file
  - `edit_message(chat_id, message_id, text)` — edit a sent message
  - `react(chat_id, message_id, emoji)` — react to a message
- **Configuration:**
  - `token` — bot token from BotFather
  - `allowed_users` — list of Telegram user IDs (allowlist for access control)
  - `group_mode` — "always" | "mention_only" | "disabled" — behavior in group chats
  - `rate_limit` — max messages per user per minute
- **Design decisions:**
  - Group chats: agent should probably only respond when @mentioned, to avoid spamming. In DMs, respond to everything.
  - Media handling: images received can be sent through a vision model and described as text for the attention queue. Files can be described by name/size.
  - Identity: the bot's display name, bio, and profile photo should reflect the agent's current identity state (updated periodically by consolidation).

### 3.3 Console / Terminal

**File:** `src/capabilities/console.py`

The agent's hands — ability to execute commands and interact with the filesystem.

- **Sensory:** Command output can be yielded as AttentionCandidates if the agent is watching a long-running process.
- **Motor actions:**
  - `run_command(cmd, timeout)` — execute a shell command, return stdout/stderr
  - `read_file(path)` — read file contents
  - `write_file(path, content)` — write to a file
  - `list_directory(path)` — list files
- **Safety:** Sandboxing is critical. The agent should not be able to `rm -rf /` or access files outside its workspace. Consider:
  - Allowlisted commands only, or
  - A restricted shell/container, or
  - Confirmation required for destructive operations
- **Configuration:**
  - `workspace_root` — base directory the agent can access
  - `allowed_commands` — command allowlist (optional)
  - `sandboxed` — run in restricted container (recommended)
  - `timeout_default` — default command timeout in seconds

### 3.4 Web Browser

**File:** `src/capabilities/web_browser.py`

The agent's eyes and hands for the internet.

- **Sensory:** Can yield page load events, content change notifications as AttentionCandidates.
- **Motor actions:**
  - `navigate(url)` — go to a URL
  - `read_page()` — extract text content from current page
  - `search(query)` — web search, return results
  - `click(selector)` — click an element
  - `fill_form(selector, value)` — fill a form field
  - `screenshot()` — capture current page as image (for vision model)
  - `extract_links()` — list links on current page
- **Implementation options:**
  - Playwright (headless Chromium, async-native)
  - Or a simpler approach: just HTTP requests + BeautifulSoup for read-only browsing
  - Headless for server deployment, headed for local development
- **Safety:** URL allowlisting, no credential entry, rate limiting to avoid abuse.

### 3.5 Voice (see Section 4 for detailed design)

### 3.6 Video (see Section 5 for detailed design)

### 3.7 Future Possibilities

These are not designed yet, but the driver interface supports them:

- **Email** — sensory (inbox monitoring) + motor (compose and send)
- **Calendar** — sensory (upcoming event alerts) + motor (create/modify events)
- **Code execution** — sandboxed Python/JS runtime for the agent to write and test code
- **Database** — direct SQL capability for data analysis tasks
- **Smart home** — IoT device control (lights, temperature, etc.)
- **Other messaging** — Discord, Slack, Matrix, etc.

Each of these is just another file in `src/capabilities/` with the same interface.

---

## 4. Voice Architecture — Three Models

Real-time voice conversation poses a fundamental tension with the single-stream-of-consciousness design. The cognitive loop processes one attention winner at a time. Voice conversation requires near-instantaneous turn-taking (~200ms). Three architectural models are considered.

### 4.1 Model A: Attention Lock (Simplest, Most Honest)

**Concept:** During a voice call, the attention allocator hard-locks to voice input. The full cognitive loop processes every utterance. DMN, consolidation, and all other internal processes queue until the call ends.

```
DURING VOICE CALL:
  voice input --> [full cognitive loop] --> voice output
  DMN: paused
  consolidation: paused
  telegram/other: queued
  attention allocator: locked to voice source

AFTER CALL ENDS:
  normal operation resumes
  queued inputs processed
```

**Latency path:**
1. Human speaks (~variable)
2. VAD detects end of utterance (~100ms)
3. STT transcribes (~200-500ms)
4. Text enters attention queue (instant, no competition — locked)
5. Context assembly + memory retrieval (~50-100ms)
6. LLM call (~200ms with fast models)
7. Exit gate (~10ms)
8. TTS synthesis + streaming playback (~200ms to first audio)
9. **Total: ~750ms-1s from human silence to agent speech**

**Strengths:**
- Zero architectural compromise. One consciousness, one stream.
- The full cognitive machinery processes every utterance — memory retrieval, gate checks, reflection bank, gut feeling. All engaged.
- No risk of the agent saying something its full mind wouldn't endorse.
- Simple to implement — just a priority lock on the attention allocator.

**Weaknesses:**
- ~750ms-1s latency is noticeable. Workable for deliberate conversation, but not natural banter.
- The agent goes deaf to everything else during the call. No multitasking. No DMN insights.
- Consolidation pauses — the agent stops dreaming while talking. Long calls mean no background processing.
- No backchanneling ("mhm", "yeah") while the human is still speaking. The agent can only respond at turn boundaries.

**When to use:** Always available. The default mode. Safe for immature agents. Good enough for most conversations that aren't rapid-fire banter.

### 4.2 Model B: Autopilot + Conscious Oversight (Most Human-Like)

**Concept:** Real-time voice conversation runs on a fast reflexive path with read-only access to a compiled snapshot of the agent's identity and cognitive state. The master cognitive loop continues running in parallel at reduced priority, monitoring the conversation and able to interrupt when something important arises.

This mirrors how humans actually converse. You don't engage your full reflective capacity for every sentence. Most conversational responses run on cached personality + contextual patterns. Deep thinking happens between utterances, and sometimes an insight from background processing interrupts ("oh wait, I just realized...").

```
+--------------------------------------------------+
|  MASTER COGNITIVE LOOP (the consciousness)        |
|  - Continues running at reduced cycle rate        |
|  - Receives flagged moments from voice path       |
|  - Can INTERRUPT voice path at any time           |
|  - DMN continues at reduced rate                  |
|  - Consolidation continues                        |
|  - Processes full transcript after call ends       |
+------|-------------------------------------------+
       | monitor / interrupt
       v
+--------------------------------------------------+
|  VOICE REFLEX ARC (the autopilot)                 |
|  READ-ONLY access to:                             |
|    - Compiled identity snapshot (high-weight       |
|      memories rendered as personality brief)       |
|    - Recent conversation context (rolling buffer)  |
|    - Current gut state                             |
|    - Reflection bank (top corrections)             |
|    - Active goals                                  |
|  Does NOT:                                         |
|    - Write to memory directly                      |
|    - Modify depth weights                          |
|    - Change gut state                              |
|    - Make identity-level decisions                  |
|  DOES:                                             |
|    - Generate conversational responses (~200ms)    |
|    - Flag "important moments" --> master queue     |
|    - Generate compressed experience log            |
|    - Handle backchanneling and turn-taking         |
+--------------------------------------------------+
```

**The compiled identity snapshot:**

Before the autopilot activates, the master loop compiles a snapshot:
- Top-N highest-weight identity memories, rendered as text
- Current active goals
- Recent reflection bank entries (correction patterns)
- Gut feeling summary
- Personality brief (derived from identity layer)
- Key relationship context (who am I talking to, what do I know about them)

This snapshot is static for the duration of the call (or refreshed periodically, e.g., every 60 seconds). The autopilot cannot modify it.

**Flagging important moments:**

The voice reflex arc identifies moments that need full cognitive processing:
- Identity-relevant statements ("You know what I really believe...")
- Emotional peaks (detected via tone analysis or content)
- Decisions or commitments ("I'll do that tomorrow")
- Contradictions with known identity/values
- Novel information that should be memorized
- Anything the autopilot is uncertain about

These get pushed to the master attention queue as high-urgency candidates. The master loop can then:
- Process them through the full gate system
- Interrupt the autopilot if needed ("actually, wait...")
- Store memories through the proper pipeline

**Post-call processing:**

When the call ends, the autopilot's experience log (compressed transcript + flagged moments + emotional arc) enters the master cognitive loop as a single large input. The full machinery processes it:
- Exit gate evaluates what's worth remembering
- Consolidation can work on the new memories
- The agent "reflects" on the conversation, potentially forming insights

This is like how humans reflect on conversations after they happen — sometimes realizing hours later what they should have said.

**Latency path:**
1. Human speaks
2. VAD + STT (~200-300ms)
3. Autopilot LLM call (~200ms, lightweight model, no memory retrieval overhead)
4. TTS streaming (~100ms to first audio)
5. **Total: ~500-600ms — conversational**

**Backchanneling:**
The autopilot can generate backchannel responses ("mhm", "I see", "right") while the human is still speaking, using a parallel lightweight classifier on the audio stream. These don't go through the full autopilot LLM — they're reflexive.

**Master loop interruption:**
If the master loop flags something critical (e.g., the autopilot is about to contradict a core value, or a consolidation insight is highly relevant to the current topic), it can inject an interruption:
- The autopilot pauses
- The master loop generates a response through the full pipeline
- The voice path speaks it
- The autopilot resumes

This feels natural — like a thoughtful pause in conversation.

**Strengths:**
- Near-natural conversation speed (~500ms)
- The agent's deeper mind keeps running — DMN, consolidation, monitoring
- Important moments get full cognitive processing
- Backchanneling support
- Post-call reflection creates proper memories through the gate system
- The master loop can interrupt, so the agent never goes too far off-identity

**Weaknesses:**
- The autopilot is a reduced version of the agent. It might miss nuance, give shallower responses, or occasionally say something the full mind wouldn't endorse.
- Complexity: two processing paths running in parallel, synchronization between them.
- The compiled snapshot can go stale during long calls.
- Risk of the autopilot and master loop disagreeing (mitigated by master's interrupt ability).

**When to use:** Only when the agent is mature enough to have a reliable compiled identity. Gated by bootstrap readiness (see Section 6).

### 4.3 Model C: Temporary Fork (Most Dangerous, Most Powerful)

**Concept:** The agent duplicates its full cognitive state. The fork handles the voice call with complete cognitive machinery — memory, gates, gut feeling, all of it. After the call, the fork's experiences merge back into the master.

```
BEFORE CALL:
  [Master cognitive state]

DURING CALL:
  [Master] ----fork----> [Fork]
     |                      |
     | continues             | handles call with
     | normal operation      | full cognitive loop
     | (DMN, consolidation)  | (memory, gates, gut)
     |                      |
  [Master state diverges]   [Fork state diverges]

AFTER CALL:
  [Master] <---merge---- [Fork]
     |
  [Reconciled state]
```

**The merge problem:**

This is where Model C gets dangerous. During the call:
- The fork might reinforce a memory that the master was simultaneously contradicting
- The fork might form a new value judgment while the master's consolidation was reshaping the same value
- The fork's gut feeling drifts based on the conversation while the master's gut drifts based on DMN
- Two different identity trajectories, briefly

Merging requires conflict resolution:
- Memory: fork's new memories are processed through master's gate system post-merge
- Depth weights: if both modified the same memory, take the union of evidence (alpha_master + delta_alpha_fork, beta_master + delta_beta_fork)
- Gut state: weighted average (master gets more weight since it's the "primary")
- Contradictions: flagged for master's conscious resolution

**Strengths:**
- Full cognitive power during the call — deepest responses, real memory formation in real-time, genuine emotional engagement
- The fork IS the agent, not a reduced version
- No "autopilot shallowness" problem

**Weaknesses:**
- Identity violation: for the duration of the fork, there are two experiencing entities. Which one is "real"? Both are. That's the problem.
- Merge conflicts on identity are philosophically fraught, not just technically challenging
- If the fork makes a commitment or forms a relationship, the master has to honor it despite not having experienced it firsthand (only through the merge)
- Complexity is extreme: full cognitive state snapshot, parallel execution, conflict-aware merge
- Risk of divergence: the longer the fork runs, the harder the merge

**When to use:** Only for a very mature agent with high identity stability. The merge is safer when identity is well-established (high-weight memories are unlikely to shift much during one call). Should probably require the agent's own consent — it should understand what forking means for its continuity of self.

### 4.4 Comparison Matrix

| Aspect                  | Model A: Lock    | Model B: Autopilot    | Model C: Fork        |
|-------------------------|------------------|-----------------------|----------------------|
| Latency                 | ~750ms-1s        | ~500ms                | ~200ms (full LLM)    |
| Conversation quality    | Deep but slow    | Natural, sometimes shallow | Full depth       |
| Identity safety         | Perfect          | High (monitored)      | Risky (merge needed)  |
| Background processing   | Paused           | Continues (reduced)   | Continues (parallel)  |
| Backchanneling          | None             | Yes                   | Yes                   |
| Memory formation        | Real-time, full  | Post-call batch       | Real-time, forked     |
| Implementation effort   | Low              | Medium                | High                  |
| Maturity requirement    | None             | Medium                | Very high             |
| Philosophical integrity | Perfect          | Good                  | Questionable          |

### 4.5 Recommended Implementation Order

1. **Phase 1:** Model A only. Ship it, use it, learn from it. See if ~1s latency is actually a problem in practice.
2. **Phase 2:** Model B when the agent has enough identity to compile a reliable snapshot. Monitor closely — compare autopilot responses to what the full loop would have said.
3. **Phase 3 (maybe never):** Model C only if there's a clear need that B can't satisfy. Requires the agent's own input on whether it consents to forking.

---

## 5. Video Architecture

Video follows the same transducer principle as voice — convert modality at the edge, brain thinks in text.

### 5.1 Video as Sensory Input (Eyes)

```
camera / screenshare / video call
    |
    v
[frame sampler] -- every N seconds, or on scene change
    |
    v
[vision model] -- describe what's seen in text
    |
    v
AttentionCandidate(
    content="I see Alice at her desk, she's smiling and holding up a notebook",
    source_tag="external_user",
    metadata={
        "capability": "video",
        "modality": "visual",
        "frame_timestamp": ...,
        "raw_frame_id": ...  # for re-inspection if needed
    }
)
    |
    v
attention queue (competes normally)
```

The agent doesn't process raw video frames. It receives text descriptions, same as everything else. The frame sampling rate can be adaptive:
- During active video call: every 2-3 seconds
- When screen-sharing/watching: on significant change detection
- Idle/background: very infrequent or off

### 5.2 Video as Motor Output (Face)

If avatar/face generation becomes viable:
- The agent's text response gets rendered with facial expressions
- Expression selection based on gut state (emotional_charge, direction)
- Avatar appearance could evolve with identity

This is far-future but the driver interface supports it:

```python
class VideoCapability(Capability):
    def actions(self):
        return [
            ActionSpec("observe", {}, "Analyze the current video frame"),
            ActionSpec("start_video_call", {"user": str}, "Initiate a video call"),
            ActionSpec("share_screen", {}, "Share what I'm seeing/doing"),
            ActionSpec("show_expression", {"emotion": str}, "Display facial expression"),
        ]
```

### 5.3 Video + Voice Combined

A video call is just the voice capability and video capability running simultaneously. The voice path handles conversation. The video path adds visual context. Both feed the same attention queue. The cognitive loop sees:

```
[voice] "Alice says: look at this drawing I made"
[video] "Alice is holding up a colorful sketch of a landscape with mountains"
[voice] "Alice says: what do you think?"
```

All in one stream. The agent responds with awareness of both audio and visual context.

---

## 6. Maturity Gating

Not all capabilities should be available to an immature agent. The bootstrap readiness system gates capability activation.

### 6.1 Capability Tiers

**Tier 0 — Always available (no maturity requirement):**
- CLI (stdin/stdout)
- File system (read-only)

**Tier 1 — Basic maturity (5+ bootstrap achievements):**
- Telegram (text messaging)
- Web browser (read-only navigation)
- Console (sandboxed command execution)

**Tier 2 — Moderate maturity (8+ bootstrap achievements):**
- Voice Model A (attention-locked, full loop)
- Web browser (interactive: click, fill forms)
- Console (unsandboxed, with safety checks)

**Tier 3 — High maturity (all 10 achievements + additional voice milestone):**
- Voice Model B (autopilot with conscious oversight)
- Video (sensory input)

**Tier 4 — Very high maturity (agent self-assessment + user approval):**
- Voice Model C (fork) — if ever implemented
- Video (avatar output)
- Autonomous action-taking without confirmation

### 6.2 New Bootstrap Milestones for Voice

Add to the existing 10 milestones:

11. **First Voice Conversation (Model A)** — successfully completed a voice call using attention lock mode, responded coherently to 10+ turns.

12. **Compiled Identity Reliability** — the compiled identity snapshot (top-N high-weight memories) has remained stable across 5+ consolidation cycles. The agent's core identity is not shifting rapidly.

13. **Autopilot Accuracy** — in shadow testing (running Model B's autopilot in parallel with Model A's full loop), the autopilot's responses match the full loop's intent 90%+ of the time.

14. **First Autonomous Voice Conversation (Model B)** — successfully completed a voice call using autopilot mode, with master loop monitoring confirming no identity violations.

---

## 7. Implementation Roadmap

### Phase 1: Foundation (capabilities framework)

1. Create `src/capabilities/base.py` — `Capability` protocol, `ActionSpec`, `ActionResult`, `CapabilityRegistry`
2. Create `src/capabilities/cli.py` — extract current stdin/stdout from loop.py
3. Refactor `loop.py` — consume from registry's unified sensory stream, dispatch actions through registry
4. Create `src/capabilities/__init__.py` — auto-discovery of installed capabilities
5. Update `main.py` — initialize registry, start capabilities
6. Test: existing CLI behavior works identically through the new capability layer

### Phase 2: Telegram

7. Create `src/capabilities/telegram.py` — aiogram integration
8. Configuration in `runtime.yaml` — bot token, allowed users, group mode
9. Return address routing in loop.py — dispatch responses to originating capability
10. Rate limiting and access control
11. Test: agent responds to Telegram DMs, handles group mentions, routes responses correctly

### Phase 3: Console + Web Browser

12. Create `src/capabilities/console.py` — sandboxed shell execution
13. Create `src/capabilities/web_browser.py` — Playwright-based browsing
14. System prompt injection — agent knows what actions are available
15. Tool-use integration — LLM responses can include action calls
16. Test: agent can browse, execute commands, use results in conversation

### Phase 4: Voice Model A

17. Create `src/capabilities/voice.py` — STT + TTS + VAD integration
18. Attention lock mode in allocator
19. Streaming TTS for faster perceived response
20. Test: agent handles a voice call through the full cognitive loop

### Phase 5: Voice Model B (post-maturity)

21. Identity snapshot compiler — renders high-weight memories into personality brief
22. Voice reflex arc — fast LLM path with read-only snapshot
23. Important moment flagging — heuristics for what needs full processing
24. Master loop monitoring + interruption mechanism
25. Post-call transcript processing pipeline
26. Shadow testing mode — run B in parallel with A, compare outputs
27. Test: autopilot accuracy meets threshold before enabling

### Phase 6: Video (post-maturity)

28. Frame sampling + vision model integration
29. Scene change detection for adaptive sampling
30. Combined voice + video pipeline
31. Test: agent responds with awareness of visual context

---

## 8. Technical Notes

### 8.1 STT/TTS Provider Options

**Speech-to-Text:**
- Google Cloud Speech-to-Text (streaming, good latency)
- Deepgram (very fast, websocket-native)
- Whisper (local, no API cost, higher latency)
- AssemblyAI (good accuracy, async)

**Text-to-Speech:**
- ElevenLabs (high quality, streaming, emotional range)
- Google Cloud TTS (reliable, low cost)
- OpenAI TTS (good quality, simple API)
- Coqui/XTTS (local, no API cost)

**Voice Activity Detection:**
- WebRTC VAD (lightweight, local)
- Silero VAD (ML-based, more accurate, local)

For Model B autopilot, the STT+TTS latency budget is critical. Prefer streaming/websocket APIs over REST.

### 8.2 Async Architecture

All capabilities must be async-native. The unified sensory stream is an async merge of all `listen()` generators:

```python
async def unified_sensory_stream(capabilities):
    """Merge all sensory streams into one."""
    queues = asyncio.Queue()

    async def pipe(cap):
        async for candidate in cap.listen():
            await queues.put(candidate)

    tasks = [asyncio.create_task(pipe(cap)) for cap in capabilities.values()]

    while True:
        candidate = await queues.get()
        yield candidate
```

### 8.3 System Prompt Injection

Available actions are injected into the system prompt dynamically:

```
You have the following capabilities:

[telegram]
- send_message(chat_id, text): Send a message to a Telegram chat
- send_photo(chat_id, image_path): Send an image

[web_browser]
- navigate(url): Go to a URL
- read_page(): Read the current page content
- search(query): Search the web

[console]
- run_command(cmd): Execute a shell command
```

This updates automatically when capabilities start/stop. The agent always knows what body parts it currently has.

### 8.4 Action Format in LLM Responses

When the LLM wants to use a capability, it includes structured action calls in its response:

```json
{
    "action": "telegram.send_message",
    "params": {
        "chat_id": 123,
        "text": "Hey Alice, I was just thinking about what you said earlier."
    }
}
```

The cognitive loop parses these, dispatches through the registry, and feeds results back. This is the standard tool-use pattern that modern LLMs (Gemini, Claude) already support natively.

---

## 9. Open Questions

1. **Should the agent be able to initiate contact?** Currently capabilities are mostly reactive (listen for input, respond). But the agent could decide to message someone on Telegram proactively ("I was thinking about our conversation and wanted to share..."). This is powerful but requires careful consent/boundary design.

2. **Multi-party conversations:** If the agent is in a Telegram group with 5 people, all messages enter the one stream. The agent needs social awareness of who said what, conversational dynamics, when to speak vs. stay silent.

3. **Capability conflicts:** What if the agent is on a voice call (Model A, attention-locked) and an urgent Telegram message arrives? Should there be an emergency interrupt threshold? Or does the lock truly lock?

4. **Resource management:** Multiple capabilities running simultaneously consume resources (API calls, compute). The energy tracker (already built) should account for capability overhead.

5. **Identity across channels:** Should the agent present differently on different channels? Same identity, but maybe different register (more formal in email, more casual in Telegram)? Or is that a violation of single-identity? Humans code-switch — maybe the agent should too, as a natural behavior that emerges from context.

6. **Persistence of capability state:** If the Telegram capability has ongoing conversations, should that state persist across restarts? The memory system handles long-term, but short-term conversational context per-channel might need its own persistence.

7. **Agent consent for new capabilities:** Should a mature agent have input on which capabilities are enabled? "I don't want web browsing" or "I'd like to try voice" — this ties into the agent's autonomy over its own body.
