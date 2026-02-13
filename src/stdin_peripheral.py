"""Stdin Peripheral — local console input/output.

The original input method, factored into a peripheral that pushes
AttentionCandidate objects into the unified input queue.

Uses a single persistent reader thread to avoid thread pool exhaustion
when stdin is a TTY (e.g. Docker with tty: true).
"""

import asyncio
import logging
import sys
import threading

from .attention import AttentionCandidate

logger = logging.getLogger("agent.stdin")


class StdinPeripheral:
    """Console stdin/stdout peripheral."""

    def __init__(self, input_queue: asyncio.Queue):
        self.input_queue = input_queue

    async def run(self, shutdown_event: asyncio.Event):
        """Read stdin in a loop, push candidates into the input queue."""
        if not sys.stdin.isatty():
            logger.info("Stdin is not a TTY — stdin peripheral disabled.")
            return

        logger.info("Stdin peripheral started.")

        # Use a dedicated thread + asyncio.Queue to avoid thread pool exhaustion.
        # A single thread runs readline() in a blocking loop, putting results
        # into a bridging queue that the async side reads from.
        bridge: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _reader_thread():
            """Blocking reader thread — runs until stdin closes or EOF."""
            while not shutdown_event.is_set():
                try:
                    line = sys.stdin.readline()
                    if not line:  # EOF
                        break
                    loop.call_soon_threadsafe(bridge.put_nowait, line)
                except Exception:
                    break

        reader = threading.Thread(target=_reader_thread, daemon=True)
        reader.start()

        while not shutdown_event.is_set():
            print("you> ", end="", flush=True)
            try:
                line = await asyncio.wait_for(bridge.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue

            text = line.strip() if line else None
            if not text:
                continue

            candidate = AttentionCandidate(
                content=text,
                source_tag="external_user",
                metadata={"reply_fn": _make_print_reply(), "peripheral": "stdin"},
            )

            try:
                self.input_queue.put_nowait(candidate)
            except asyncio.QueueFull:
                print("[Input queue full — try again]")

        logger.info("Stdin peripheral stopped.")


def _make_print_reply():
    """Create a reply callback that prints to stdout."""

    async def reply_fn(text: str):
        print(f"\n{text}\n")

    return reply_fn
