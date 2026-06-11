"""Mixin for asynchronous experience generation in RL training."""

import logging
import queue
import threading
from typing import (
    Any,
)

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """A bounded queue for pre-computed RL experience."""

    def __init__(self, max_size: int = 1):
        self.queue = queue.Queue(maxsize=max_size)

    def put(self, experience: dict[str, Any]):
        self.queue.put(experience)

    def get(self) -> dict[str, Any]:
        return self.queue.get()


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

        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None

    def start_producer(
        self, model, ref_model, data_iter, generation_engine, reward_fn, collective
    ):
        """Start the background producer thread (Rollout group)."""
        if self.role != "rollout":
            return

        def producer_loop():
            while not self._stop_event.is_set():
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
                    self.buffer.put(experience)
                except Exception as e:
                    logger.error(f"Error in producer loop: {e}")
                    break

        self._producer_thread = threading.Thread(target=producer_loop, daemon=True)
        self._producer_thread.start()

    def stop_producer(self):
        """Stop the background producer thread."""
        self._stop_event.set()
        if self._producer_thread:
            self._producer_thread.join()

    def _produce_one_batch(
        self, model, ref_model, data_iter, generation_engine, reward_fn, collective
    ) -> dict[str, Any]:
        """Generate one batch of experience (Rollout group logic)."""
        # This implementation will be refined in the GRPORecipe
        raise NotImplementedError()

    def get_batch(self) -> dict[str, Any]:
        """Get a batch of experience (Trainer group logic)."""
        return self.buffer.get()
