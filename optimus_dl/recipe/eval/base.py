"""Evaluation recipe for LLM Baselines models."""

import logging
from pathlib import Path

import torch

from optimus_dl.modules.checkpoint import CheckpointManager
from optimus_dl.modules.distributed import build_best_collective
from optimus_dl.modules.distributed.base import Collective
from optimus_dl.modules.eval import LLMBaselinesModel
from optimus_dl.modules.tokenizer import build_tokenizer  # New import
from optimus_dl.recipe.eval.config import EvalConfig
from optimus_dl.recipe.mixins import ModelBuilder

logger = logging.getLogger(__name__)


class EvalRecipe:
    """Recipe for evaluating LLM Baselines models using lm_eval harness."""

    def __init__(self, cfg: EvalConfig):
        """Initialize evaluation recipe.

        Args:
            cfg: Evaluation configuration
        """
        self.cfg = cfg
        self.model = None
        self.tokenizer = None

        # Initialize builders using composition
        # ModelBuilder needs a dummy config, but it won't be used since we use build_model_from_checkpoint
        self.model_builder = ModelBuilder(None, [])
        self.checkpoint_manager = CheckpointManager(None)

        # Direct tokenizer build config
        self.tokenizer_config = cfg.common.tokenizer

    def build_eval_model(self, collective: Collective) -> LLMBaselinesModel:
        """Build and load model from checkpoint for evaluation.

        Returns:
            Loaded LLMBaselinesModel instance
        """
        if self.model is None:
            assert (self.cfg.common.checkpoint_path is not None) ^ (
                self.cfg.common.model is not None
            ), "Either checkpoint_path or model must be specified, but not both"

            device = collective.default_device
            if self.cfg.common.checkpoint_path is not None:
                logger.info(
                    f"Loading model from checkpoint: {self.cfg.common.checkpoint_path}"
                )
                base_model, _ = self.ch.build_model_from_checkpoint(
                    checkpoint_path=self.cfg.common.checkpoint_path, device=device
                )
            else:
                logger.info("Building model from config")
                base_model = self.model_builder.build_model(
                    model_config=self.cfg.common.model,
                    collective=collective,
                )

            # Build tokenizer directly
            self.tokenizer = build_tokenizer(self.tokenizer_config)

            # Wrap in LLMBaselinesModel for lm_eval compatibility
            base_model.eval()
            self.model = LLMBaselinesModel(
                model=base_model.to(device),
                tokenizer=self.tokenizer,
                tokenizer_config=self.tokenizer_config,
                device=device,
            )

        return self.model

    def run_lm_eval(self) -> dict:
        """Run lm_eval harness evaluation.

        Returns:
            Dictionary with evaluation results
        """
        try:
            from lm_eval import evaluator
        except ImportError as err:
            raise ImportError(
                "lm_eval is required for evaluation. Install with: pip install lm_eval"
            ) from err

        # Build model
        collective = build_best_collective(
            device=None if self.cfg.common.use_gpu else torch.device("cpu"),
            tp_size=1,
        )
        model = self.build_eval_model(collective=collective)

        logger.info(f"Running lm_eval on tasks: {self.cfg.lm_eval.tasks}")
        logger.info(f"Few-shot examples: {self.cfg.lm_eval.num_fewshot}")

        # Convert tasks to proper format for lm_eval
        raw_tasks = self.cfg.lm_eval.tasks
        if isinstance(raw_tasks, str):
            tasks = [raw_tasks]
        else:
            # Convert to list of strings, handling any nested structure
            tasks = []
            for task in raw_tasks:
                if isinstance(task, str):
                    tasks.append(task)
                else:
                    tasks.append(str(task))

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=self.cfg.lm_eval.num_fewshot,
            batch_size=self.cfg.lm_eval.batch_size,
            limit=self.cfg.lm_eval.limit,
            device=collective.default_device,
            use_cache=None,  # Disable caching for now
            # verbosity="INFO",
        )

        if results is None:
            raise RuntimeError("Evaluation returned no results")

        # Save results if output path is specified
        if self.cfg.lm_eval.output_path:
            import json

            output_path = Path(self.cfg.lm_eval.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")

        return results
