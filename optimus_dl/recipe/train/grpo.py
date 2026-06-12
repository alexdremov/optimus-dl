"""GRPO (Group Relative Policy Optimization) training recipe."""

import gc
import logging
from dataclasses import (
    dataclass,
    field,
)
from typing import Any

import torch

from optimus_dl.core.generation import GenerationConfig
from optimus_dl.core.log import trange
from optimus_dl.core.registry import build as build_component
from optimus_dl.modules.loggers import RunStatus
from optimus_dl.modules.metrics import (
    compute_meters,
    log_averaged,
    log_event_end,
    log_event_start,
    meters_group,
)
from optimus_dl.recipe.train.base import TrainRecipe
from optimus_dl.recipe.train.config import TrainConfig
from optimus_dl.recipe.train.mixins.rl_async import AsyncExperienceManager

from . import register_train_recipe

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig(TrainConfig):
    """Configuration for GRPO Training.

    Attributes:
        num_generations: Number of completions sampled per prompt (G in GRPO paper).
        generation_engine: Config for the generation engine (e.g. native, vLLM).
        generation_config: Decoding hyper-parameters (temperature, top-k, max tokens, …).
        reward_functions: Non-empty list of reward-function configs.
        ref_model_transforms: Model transforms for the frozen reference model.
            Defaults to the same transforms as the policy when ``None``.
        tokenizer_config: Explicit tokenizer config for decoding token IDs during
            reward computation.  If ``None``, the recipe falls back to inspecting
            the data-pipeline transform chain for a ``tokenize`` step.
    """

    num_generations: int = 8
    generation_engine: Any = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    reward_functions: list[Any] = field(default_factory=list)
    ref_model_transforms: Any = None
    tokenizer_config: Any = None


@register_train_recipe("grpo", GRPOConfig)
class GRPORecipe(TrainRecipe, AsyncExperienceManager):
    """Recipe for GRPO training.

    Supports a *unified* mode where every rank hosts both the policy and the
    frozen reference model and runs rollout + optimisation synchronously.
    A partitioned (async rollout-worker / trainer-worker) mode is scaffolded
    for future extension but is not yet implemented.
    """

    cfg: GRPOConfig

    def __init__(self, cfg: GRPOConfig) -> None:
        # Initialise AsyncExperienceManager first so self.role is available
        # before TrainRecipe.__init__ (which calls validate_config).
        AsyncExperienceManager.__init__(self, role="unknown")
        TrainRecipe.__init__(self, cfg)

    # ------------------------------------------------------------------ #
    # Config validation                                                    #
    # ------------------------------------------------------------------ #

    def validate_config(self) -> None:
        """Extend base validation with GRPO-specific checks."""
        super().validate_config()
        assert (
            self.cfg.reward_functions
        ), "reward_functions must be a non-empty list of reward-function configs"
        assert (
            self.cfg.generation_engine is not None
        ), "generation_engine config is required for GRPO"

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> None:  # noqa: C901 (complexity is unavoidable here)
        """Run the complete GRPO pipeline."""
        self.setup_context()
        is_restart = self.checkpoint_manager.is_restart()
        finished_run = False

        with meters_group("init"):
            log_event_start("perf/init")
            logger.info(f"Using output path: {self.cfg.common.output_path}")
            logger.info(self.cfg)

            # 1. Device and distributed collective
            device, collective, logs_parent_path = self._setup_distributed()

            # Determine per-rank role
            self.role = self._determine_role(collective)
            logger.info(f"Rank {collective.rank} assigned role: {self.role}")

            if self.role == "idle":
                logger.info("Rank idle, waiting for others...")
                collective.barrier()
                return

            # 2. Policy model
            logger.info(f"Building Policy model on {device}...")
            model = self.build_model(
                model_config=self.cfg.model,
                collective=collective,
                is_restart=is_restart,
                checkpoint_manager=self.checkpoint_manager,
            ).to(device)

            # 3. Frozen reference model (only needed on rollout / unified ranks)
            ref_model = None
            if self.role in ("rollout", "unified"):
                logger.info("Building Reference model...")
                ref_transforms = (
                    self.cfg.ref_model_transforms or self.cfg.model_transforms
                )
                ref_model = self.model_builder.build_model(
                    model_config=self.cfg.model,
                    collective=collective,
                    model_transforms=ref_transforms,
                ).to(device)
                ref_model.eval()
                for p in ref_model.parameters():
                    p.requires_grad_(False)

            # 4. Training components (trainer / unified ranks only)
            optimizer = lr_scheduler = criterion = None
            if self.role in ("trainer", "unified"):
                optimizer = self.build_optimizer(model.make_parameter_groups())
                criterion = self.build_criterion(collective=collective)
                lr_scheduler = self.build_lr_scheduler(optimizer)

            training_context = self.setup_training_context(device)

            # 5. Data pipelines
            try:
                train_datapipeline = self.build_train_data(
                    device=device, collective=collective
                )
                assert (
                    train_datapipeline is not None
                ), "Train data pipeline not initialized"
                eval_datapipeline = self.build_eval_data(
                    device=device, collective=collective
                )
            except Exception as e:
                logger.error(f"Failed to build data pipelines: {e}")
                raise

            # 6. Tokenizer for reward text decoding
            self.tokenizer = self._build_tokenizer()

            # 7. Checkpoint resume (trainer / unified ranks only)
            common_chkp_kwargs: dict[str, Any] = {}
            start_iteration = 0
            metadata = None
            if self.role in ("trainer", "unified"):
                data_loaders = {"train": train_datapipeline.dataloader}
                common_chkp_kwargs = {
                    "model": model,
                    "optimizer": optimizer,
                    "collective": collective,
                    "lr_scheduler": lr_scheduler,
                    "data_loaders": data_loaders,
                    "data_sources": train_datapipeline.datasets,
                    "grad_scaler": training_context["scaler"],
                }
                start_iteration, metadata, finished_run, common_chkp_kwargs = (
                    self._init_checkpointing(
                        model,
                        optimizer,
                        lr_scheduler,
                        training_context,
                        train_datapipeline,
                        collective,
                        is_restart,
                    )
                )

            # 8. Loggers (master only)
            if self.role in ("trainer", "unified"):
                self._init_loggers(collective, logs_parent_path, start_iteration)

            log_event_end("perf/init")

        # Log init metrics
        init_metrics = compute_meters("init", aggregate=True, collective=collective)
        if collective.is_local_master:
            self.log_metrics_to_loggers(init_metrics, start_iteration, "init")

        # 9. Build generation engine and reward functions
        gen_engine = build_component("generation_engine", self.cfg.generation_engine)
        reward_fns = [
            build_component("reward_function", r) for r in self.cfg.reward_functions
        ]

        # 10. Pre-training barrier — all ranks ready
        logger.debug("Reaching pre-training barrier...")
        collective.barrier()
        logger.info("All ranks are ready")

        # 11. Execute training loop
        try:
            if self.role == "rollout":
                self._run_rollout_worker(
                    model,
                    ref_model,
                    iter(train_datapipeline.dataloader),
                    gen_engine,
                    reward_fns,
                    collective,
                )
            elif self.role == "trainer":
                self._run_trainer_worker(
                    model,
                    optimizer,
                    criterion,
                    lr_scheduler,
                    collective,
                )
            else:
                # Unified: policy + reference on the same group of GPUs
                if not finished_run:
                    self._run_unified_loop(
                        model=model,
                        ref_model=ref_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        lr_scheduler=lr_scheduler,
                        train_data_iter=iter(train_datapipeline.dataloader),
                        eval_datapipeline=eval_datapipeline,
                        gen_engine=gen_engine,
                        reward_fns=reward_fns,
                        collective=collective,
                        device=device,
                        training_context=training_context,
                        start_iteration=start_iteration,
                        common_chkp_kwargs=common_chkp_kwargs,
                        metadata=metadata,
                    )
                else:
                    logger.info("Finished run resumed; running final evaluation only.")
                    self.evaluate_and_log(
                        iteration=start_iteration,
                        model=model,
                        criterion=criterion,
                        eval_datapipeline=eval_datapipeline,
                        collective=collective,
                        device=device,
                    )
        finally:
            if self.role in ("trainer", "unified") and collective.is_master:
                logger.debug("Closing loggers...")
                self.close_loggers()
            gc.collect()
            collective.close()
            logger.info("Training run complete")

    # ------------------------------------------------------------------ #
    # Role determination                                                   #
    # ------------------------------------------------------------------ #

    def _determine_role(self, collective) -> str:
        """Return the role string for this rank based on partition config."""
        group_cfg = self.cfg.common.distributed.partitions
        if group_cfg:
            global_rank = collective.rank
            for partition in group_cfg:
                if global_rank in partition.ranks:
                    return "trainer" if "actor" in partition.name.lower() else "rollout"
            return "idle"
        # No partitions → every rank participates in the unified loop
        return "unified"

    # ------------------------------------------------------------------ #
    # Unified synchronous loop                                             #
    # ------------------------------------------------------------------ #

    def _run_unified_loop(
        self,
        model,
        ref_model,
        optimizer,
        criterion,
        lr_scheduler,
        train_data_iter,
        eval_datapipeline,
        gen_engine,
        reward_fns,
        collective,
        device,
        training_context,
        start_iteration: int,
        common_chkp_kwargs: dict,
        metadata: dict | None,
    ) -> None:
        """Synchronous GRPO loop — Policy and Reference model co-located."""
        logger.info(f"Starting Unified GRPO Loop from iteration {start_iteration}...")

        # Build metric engine using base class method
        train_metric_engine = self._build_train_metric_engine()

        # Handle pending evaluation resumption using base class method
        if self.role in ("trainer", "unified"):
            self._handle_pending_eval_resumption(
                metadata,
                model,
                criterion,
                eval_datapipeline,
                collective,
                device,
                common_chkp_kwargs,
            )

        pbar = trange(
            start_iteration,
            self.cfg.optimization.iterations + 1,
            initial=start_iteration,
            total=self.cfg.optimization.iterations,
            miniters=self.cfg.common.log_freq,
            maxinterval=1_000_000,
            disable=not collective.is_local_master,
            smoothing=0,
        )

        for iteration in pbar:
            try:
                logger.debug(f"Starting GRPO iteration {iteration}")

                # 1. Fetch next prompt batch
                try:
                    batch = next(train_data_iter)
                except StopIteration:
                    logger.info("Data iterator exhausted.")
                    break

                # 2. Rollout: generate completions, score rewards, compute logprobs
                with meters_group("train", log_freq=self.cfg.common.log_freq):
                    experience = self._generate_experience(
                        model, ref_model, batch, gen_engine, reward_fns, device
                    )

                # 3. Policy optimisation on the generated experience.
                #    Repeat the experience tensor acc_steps times so gradient
                #    accumulation inside run_training_iteration works correctly.
                acc_steps = self.cfg.optimization.acc_steps
                self.run_training_iteration(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_data_iter=iter([experience] * acc_steps),
                    training_context=training_context,
                    lr_scheduler=lr_scheduler,
                    metric_engine=train_metric_engine,
                )

                # 4. Metrics logging, evaluation, and checkpointing
                self._log_and_checkpoint_iteration(
                    iteration=iteration,
                    pbar=pbar,
                    model=model,
                    criterion=criterion,
                    collective=collective,
                    device=device,
                    eval_datapipeline=eval_datapipeline,
                    common_chkp_kwargs=common_chkp_kwargs,
                    train_metric_engine=train_metric_engine,
                )

                logger.debug(f"Finished GRPO iteration {iteration}")

            except KeyboardInterrupt:
                logger.info("Training interrupted by user")
                self.handle_training_interruption(
                    iteration=iteration,
                    **common_chkp_kwargs,
                )
                if collective.is_master:
                    self.close_loggers(run_status=RunStatus.INTERRUPTED)
                break
            except Exception as e:
                logger.error(f"GRPO training step failed at iteration {iteration}: {e}")
                raise

    # ------------------------------------------------------------------ #
    # Partitioned-mode stubs (not yet implemented)                         #
    # ------------------------------------------------------------------ #

    def _run_rollout_worker(
        self, model, ref_model, data_iter, gen_engine, reward_fns, collective
    ) -> None:
        raise NotImplementedError(
            "Partitioned rollout-worker mode is not yet implemented. "
            "Use unified mode (common.distributed.partitions: null)."
        )

    def _run_trainer_worker(
        self, model, optimizer, criterion, lr_scheduler, collective
    ) -> None:
        raise NotImplementedError(
            "Partitioned trainer-worker mode is not yet implemented. "
            "Use unified mode (common.distributed.partitions: null)."
        )

    # ------------------------------------------------------------------ #
    # Experience generation                                                #
    # ------------------------------------------------------------------ #

    def _generate_experience(
        self,
        model,
        ref_model,
        batch: dict[str, Any],
        gen_engine,
        reward_fns: list,
        device: torch.device,
    ) -> dict[str, Any]:
        """Generate rollouts, score rewards, and compute per-token log-probs.

        Returns a dict suitable for ``GRPOCriterion.__call__`` with keys:
        ``input_ids``, ``completion_mask``, ``old_logprobs``, ``ref_logprobs``,
        ``advantages``.
        """
        prompt_ids: torch.Tensor = batch["input_ids"].to(device)
        # attention_mask marks real (non-pad) tokens; shape (B, L)
        attention_mask: torch.Tensor | None = (
            batch["attention_mask"].to(device) if "attention_mask" in batch else None
        )
        batch_size = prompt_ids.size(0)
        G = self.cfg.num_generations

        # Expand prompts: (B, L) → (B*G, L)
        expanded_prompt_ids = prompt_ids.repeat_interleave(G, dim=0)

        with torch.no_grad():
            # --- Generate completions ---
            logger.debug(
                f"Rollout: generating G={G} completions per prompt "
                f"({expanded_prompt_ids.shape[0]} total sequences)"
            )
            all_ids = gen_engine.generate(
                model, expanded_prompt_ids, self.cfg.generation_config
            )
            logger.debug(
                f"Rollout: generation complete. all_ids shape: {all_ids.shape}"
            )

            # Restore training mode; NativeEngine calls model.eval() internally.
            model.train()

            # --- Build completion mask ---
            # Respect variable prompt lengths via attention_mask when available.
            completion_mask = self._build_completion_mask(
                all_ids, prompt_ids, attention_mask, G
            )
            num_completion_tokens = completion_mask.sum().item()
            log_averaged(
                "num_completion_tokens",
                lambda: num_completion_tokens / batch_size,
                weight=batch_size,
                round=1,
            )

            # --- Reference log-probs (frozen model) ---
            logger.debug("Rollout: computing reference log-probs...")
            ref_logprobs = self._get_per_token_logprobs(
                ref_model(all_ids)["logits"], all_ids
            )

            # --- Old-policy log-probs (same model that generated) ---
            logger.debug("Rollout: computing old-policy log-probs...")
            old_logprobs = self._get_per_token_logprobs(
                model(all_ids)["logits"], all_ids
            )

            # --- Text decoding for reward functions ---
            prompts, completions, answers = self._decode_batch(
                prompt_ids,
                all_ids,
                completion_mask,
                batch,
                batch_size,
                G,
                attention_mask,
            )

            # --- Compute and aggregate rewards ---
            reward_tensors = [
                r_fn(prompts=prompts, completions=completions, answers=answers).to(
                    device
                )
                for r_fn in reward_fns
            ]
            total_rewards = torch.stack(reward_tensors).sum(dim=0)  # (B*G,)

            log_averaged(
                "reward",
                lambda: total_rewards.mean().item(),
                weight=batch_size,
                round=4,
            )

            # --- Group-relative advantage normalisation ---
            grouped_rewards = total_rewards.view(batch_size, G)
            group_mean = grouped_rewards.mean(dim=1, keepdim=True)
            group_std = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = ((grouped_rewards - group_mean) / group_std).view(-1)  # (B*G,)

            log_averaged(
                "advantage_max",
                lambda: advantages.max().item(),
                weight=batch_size,
                round=4,
            )
            log_averaged(
                "reward_std",
                lambda: grouped_rewards.std(dim=1).mean().item(),
                weight=batch_size,
                round=4,
            )

        return {
            "input_ids": all_ids,
            "completion_mask": completion_mask,
            "old_logprobs": old_logprobs,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_per_token_logprobs(
        logits: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Per-token log-probabilities for sampled tokens.

        Args:
            logits: ``(B, T, V)`` model output logits.
            tokens: ``(B, T)`` token IDs.

        Returns:
            ``(B, T-1)`` — ``log p(token_t | token_{<t})`` for every position.
        """
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        return torch.gather(
            log_probs, dim=-1, index=tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

    @staticmethod
    def _build_completion_mask(
        all_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        G: int,
    ) -> torch.Tensor:
        """Build a binary mask (0=prompt, 1=completion) for every token in ``all_ids``.

        Supports variable-length prompts via ``attention_mask`` if provided;
        falls back to the padded prompt length otherwise.
        """
        mask = torch.zeros_like(all_ids)
        if attention_mask is not None:
            # prompt_lengths[i] = number of real (non-pad) prompt tokens
            prompt_lengths = attention_mask.sum(dim=1).long()  # (B,)
            expanded_lengths = prompt_lengths.repeat_interleave(G)  # (B*G,)
            for idx, plen in enumerate(expanded_lengths.tolist()):
                mask[idx, plen:] = 1
        else:
            prompt_len = prompt_ids.size(1)
            mask[:, prompt_len:] = 1
        return mask

    def _decode_batch(
        self,
        prompt_ids: torch.Tensor,
        all_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        batch: dict[str, Any],
        batch_size: int,
        G: int,
        attention_mask: torch.Tensor | None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Decode token tensors into text strings for reward computation."""
        has_answers = "answer" in batch
        prompts: list[str] = []
        completions: list[str] = []
        answers: list[str] = []

        if self.tokenizer is not None:
            for i in range(batch_size):
                # Decode only non-padded prompt tokens
                if attention_mask is not None:
                    real_len = int(attention_mask[i].sum().item())
                    p_ids = prompt_ids[i, :real_len].tolist()
                else:
                    p_ids = prompt_ids[i].tolist()
                p_text = self.tokenizer.decode(p_ids)
                ans_text = batch["answer"][i] if has_answers else ""

                for g in range(G):
                    seq_idx = i * G + g
                    prompts.append(p_text)
                    answers.append(ans_text)
                    # Decode only completion tokens (mask == 1)
                    c_ids = all_ids[seq_idx][completion_mask[seq_idx].bool()].tolist()
                    c_text = self.tokenizer.decode(c_ids)
                    completions.append(c_text)

                    if i == 0 and g == 0:
                        logger.debug(
                            f"Rollout sample 0 — "
                            f"Prompt: {p_text[:100]!r}… "
                            f"Completion: {c_text[:100]!r}…"
                        )
        else:
            logger.warning(
                "No tokenizer available in GRPORecipe. "
                "Set `tokenizer_config` in GRPOConfig or add a `tokenize` transform "
                "to the data pipeline. Reward functions will receive empty strings."
            )
            n = batch_size * G
            prompts = [""] * n
            completions = [""] * n
            answers = [""] * n

        return prompts, completions, answers

    def _build_tokenizer(self):
        """Build tokenizer from explicit config or by inspecting the data pipeline."""
        # Explicit config takes priority
        if self.cfg.tokenizer_config is not None:
            from optimus_dl.modules.tokenizer import build_tokenizer

            return build_tokenizer(self.cfg.tokenizer_config)

        # Heuristic fallback: look for a 'tokenize' step in the transform chain
        if hasattr(self.cfg.data, "train_datasets") and hasattr(
            self.cfg.data.train_datasets, "transform"
        ):
            transform_cfg = self.cfg.data.train_datasets.transform
            if hasattr(transform_cfg, "transforms"):
                for t in transform_cfg.transforms:
                    if hasattr(t, "get") and t.get("_name") == "tokenize":
                        from optimus_dl.modules.tokenizer import build_tokenizer

                        return build_tokenizer(t.tokenizer_config)

        logger.warning(
            "No tokenizer found for GRPORecipe. "
            "Provide `tokenizer_config` in GRPOConfig or add a `tokenize` transform."
        )
        return None
