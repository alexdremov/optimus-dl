"""Mixin for asynchronous experience generation in RL training."""

import logging
import queue
import threading
from typing import (
    Any,
)

logger = logging.getLogger(__name__)

_PUT_POLL_SECONDS = 0.1


class ExperienceBuffer:
    """A bounded queue for pre-computed RL experience.

    Both ``put`` and ``get`` remain responsive to ``stop()``: they poll with a
    timeout instead of blocking forever, so a producer blocked on a full queue
    (or a consumer on an empty one) can always be shut down.
    """

    def __init__(self, max_size: int = 1):
        self.queue = queue.Queue(maxsize=max_size)
        self._stop_event = threading.Event()

    def put(self, experience: dict[str, Any]) -> bool:
        """Put an item, waiting up to the stop signal. Returns False if stopped."""
        while not self._stop_event.is_set():
            try:
                self.queue.put(experience, timeout=_PUT_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Get an item. Returns None if stopped or timed out."""
        deadline_poll = timeout if timeout is not None else _PUT_POLL_SECONDS
        while True:
            if self._stop_event.is_set():
                return None
            try:
                return self.queue.get(timeout=deadline_poll)
            except queue.Empty:
                if timeout is not None:
                    return None
                continue

    def stop(self):
        """Signal all pending put/get callers to give up."""
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()


class AsyncExperienceManager:
    """Manages the producer-consumer pipeline for RL experience.

    Attributes:
        role: Either 'trainer' or 'rollout'.
        buffer: ExperienceBuffer instance.
    """

    def __init__(
        self,
        role: str,
        buffer_size: int = 1,
        generation_config: Any | None = None,
        reward_config: Any | None = None,
    ):
        self.role = role
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.generation_config = generation_config
        self.reward_config = reward_config

        self._producer_thread: threading.Thread | None = None

    def start_producer(
        self, model, ref_model, data_iter, generation_engine, reward_fn, collective
    ):
        """Start the background producer thread (Rollout group)."""
        if self.role != "rollout":
            return

        def producer_loop():
            while not self.buffer.is_stopped():
                try:
                    # 1. Rollout
                    # 2. Score
                    # 3. Compute Ref Logprobs
                    # 4. Put in buffer (this will block if buffer is full)
                    experience = self._produce_one_batch(
                        model,
                        ref_model,
                        data_iter,
                        generation_engine,
                        reward_fn,
                        collective,
                    )
                    if not self.buffer.put(experience):
                        logger.info("Producer loop: buffer stopped; exiting.")
                        break
                except Exception as e:
                    logger.error(f"Error in producer loop: {e}")
                    break

        self._producer_thread = threading.Thread(target=producer_loop, daemon=True)
        self._producer_thread.start()

    def stop_producer(self, join_timeout: float = 5.0) -> None:
        """Stop the background producer thread.

        Signals the shared stop event first so that a producer blocked on a
        full ``buffer.put`` wakes up and exits instead of deadlocking the join.
        """
        self.buffer.stop()
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=join_timeout)
            if self._producer_thread.is_alive():
                logger.warning("Producer thread did not terminate before timeout.")
        self._producer_thread = None

    def _produce_one_batch(
        self, model, ref_model, data_iter, generation_engine, reward_fn, collective
    ) -> dict[str, Any]:
        """Generate one batch of experience (Rollout group logic)."""
        # This implementation will be refined in the GRPORecipe
        raise NotImplementedError()

    def get_batch(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Get a batch of experience (Trainer group logic)."""
        return self.buffer.get(timeout=timeout)
