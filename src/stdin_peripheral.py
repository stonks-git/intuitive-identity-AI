"""Stdin Peripheral — local console input/output.

The original input method, factored into a peripheral that pushes
AttentionCandidate objects into the unified input queue.
"""

import asyncio
import logging
import sys

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
        loop = asyncio.get_event_loop()

        while not shutdown_event.is_set():
            print("you> ", end="", flush=True)
            try:
                line = await asyncio.wait_for(
                    loop.run_in_executor(None, sys.stdin.readline),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            text = line.strip() if line else None
            if not text:
                continue

            candidate = AttentionCandidate(
                content=text,
                source_tag="external_user",
                metadata={"reply_fn": _make_print_reply()},
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
