import logging
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import torch

from optimus_dl.core.device import setup_device_and_collective
from optimus_dl.core.generation import (
    GenerationConfig,
)
from optimus_dl.core.registry import build as build_component
from optimus_dl.modules.metrics import (
    compute_meters,
    log_averaged,
    meters_group,
    reset_meters,
    step_meters,
)
from optimus_dl.recipe.train.base import TrainRecipe
from optimus_dl.recipe.train.config import (
    TrainConfig,
)
from optimus_dl.recipe.train.mixins.rl_async import AsyncExperienceManager

from . import register_train_recipe

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig(TrainConfig):
    """Configuration for GRPO Training.

    Attributes:
        num_generations: Number of generations per prompt (G).
        generation_config: Parameters for the generation engine.
        reward_config: Configuration for reward functions.
        ref_model_transforms: Transforms specifically for the reference model.
    """

    num_generations: int = 8
    generation_engine: Any = None
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    reward_functions: Any = None
    ref_model_transforms: Any = None


@register_train_recipe("grpo", GRPOConfig)
class GRPORecipe(TrainRecipe, AsyncExperienceManager):
    """Recipe for GRPO training with asynchronous rollout-optimization overlapping."""

    cfg: GRPOConfig

    def __init__(self, cfg: GRPOConfig):
        # We don't call TrainRecipe.__init__ directly yet because we need to
        # handle the dual-model setup.
        super().__init__(cfg)

        # Initialize AsyncExperienceManager
        # The role will be determined in run() after collective setup.
        AsyncExperienceManager.__init__(self, role="unknown")
        TrainRecipe.__init__(self, cfg)

    def run(self):
        """Run the GRPO pipeline."""
        self.setup_context()

        import pathlib

        from optimus_dl.core.log import setup_logging

        # 1. Setup global device and collective
        device, collective = setup_device_and_collective(
            use_gpu=self.cfg.common.use_gpu, config=self.cfg.common.distributed
        )

        # Setup console logging
        logs_parent_path = pathlib.Path(self.cfg.common.output_path) / "logging"
        rank = collective.rank if collective is not None else 0
        log_path = logs_parent_path / f"rank_{rank}"
        if not collective.is_master:
            setup_logging(logging.WARNING)
            setup_logging(log_path=log_path, clear_existing=False)
        else:
            setup_logging(log_path=log_path, clear_existing=False)

        # Determine roles. Support partitioned or unified modes.
        group_cfg = self.cfg.common.distributed.partitions
        if group_cfg:
            my_group_cfg = None
            global_rank = collective.rank
            for p in group_cfg:
                if global_rank in p.ranks:
                    my_group_cfg = p
                    break
            if my_group_cfg:
                self.role = (
                    "trainer" if "actor" in my_group_cfg.name.lower() else "rollout"
                )
            else:
                self.role = "idle"
        else:
            # Unified mode: This rank does everything (Policy + Reference)
            self.role = "unified"

        logger.info(f"Rank {collective.rank} assigned role: {self.role}")
        if self.role == "idle":
            logger.info("Rank idle, waiting for others...")
            collective.barrier()
            return

        if self.role in ["trainer", "unified"]:
            self.setup_loggers(collective)

        # 2. Build Models
        logger.info(f"Building Policy model on {device}...")
        is_restart = self.checkpoint_manager.is_restart()
        model = self.build_model(
            model_config=self.cfg.model,
            collective=collective,
            is_restart=is_restart,
            checkpoint_manager=self.checkpoint_manager,
        ).to(device)

        ref_model = None
        if self.role in ["rollout", "unified"]:
            logger.info("Building Reference model...")
            ref_transforms = self.cfg.ref_model_transforms or self.cfg.model_transforms
            ref_model = self.model_builder.build_model(
                model_config=self.cfg.model,
                collective=collective,
                model_transforms=ref_transforms,
            ).to(device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        # 3. Build Training Components
        optimizer = None
        criterion = None
        lr_scheduler = None
        if self.role in ["trainer", "unified"]:
            optimizer = self.build_optimizer(model.make_parameter_groups())
            criterion = self.build_criterion(collective=collective)
            lr_scheduler = self.build_lr_scheduler(optimizer)

        # Setup training context for scaler
        training_context = self.setup_training_context(device)

        # 4. Data and Engines
        train_datapipeline = self.build_train_data(device=device, collective=collective)
        train_data_iter = iter(train_datapipeline.dataloader)

        eval_datapipeline = self.build_eval_data(device=device, collective=collective)

        # Build tokenizer for reward calculation
        # We look for a 'tokenize' transform in the data pipeline config
        self.tokenizer = None
        if (
            hasattr(self.cfg.data, "train_datasets")
            and "transform" in self.cfg.data.train_datasets
        ):
            transform_cfg = self.cfg.data.train_datasets.transform
            # It might be a compose transform
            if hasattr(transform_cfg, "transforms"):
                for t in transform_cfg.transforms:
                    if t.get("_name") == "tokenize":
                        from optimus_dl.modules.tokenizer import build_tokenizer

                        self.tokenizer = build_tokenizer(t.tokenizer_config)
                        break

        # Checkpoints
        start_iteration = 0
        if self.role in ["trainer", "unified"]:
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
            start_iteration, metadata = self.load_checkpoint_if_exists(
                **common_chkp_kwargs
            )
            logger.info(f"Checkpointed start_iteration: {start_iteration}")
            if not is_restart and self.cfg.common.load_checkpoint is not None:
                metadata = self.load_checkpoint(
                    **common_chkp_kwargs,
                    load_strategy=self.cfg.common.load_checkpoint_strategy,
                    checkpoint_path=self.cfg.common.load_checkpoint,
                )
                start_iteration = metadata["iteration"] + 1
                logger.info(f"Loaded checkpoint start_iteration: {start_iteration}")

        if self.role in ["trainer", "unified"] and collective.is_master:
            self.build_loggers()
            self.setup_loggers(
                experiment_name=self.cfg.common.exp_name,
                logs_parent_path=logs_parent_path,
                start_iteration=start_iteration,
            )

        gen_engine = build_component("generation_engine", self.cfg.generation_engine)
        reward_fns = [
            build_component("reward_function", r) for r in self.cfg.reward_functions
        ]

        # 5. Execute Loop
        try:
            if self.role == "rollout":
                self._run_rollout_worker(
                    model,
                    ref_model,
                    train_data_iter,
                    gen_engine,
                    reward_fns,
                    collective,
                )
            elif self.role == "trainer":
                self._run_trainer_worker(
                    model, optimizer, criterion, lr_scheduler, collective
                )
            else:
                # Unified: Synchronous loop on one group of GPUs
                self._run_unified_loop(
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
                    start_iteration,
                    common_chkp_kwargs,
                )
        finally:
            if self.role in ["trainer", "unified"] and collective.is_master:
                logger.debug("Closing loggers...")
                self.close_loggers()

    def _run_unified_loop(
        self,
        model,
        ref_model,
        optimizer,
        criterion,
        lr_scheduler,
        data_iter,
        eval_datapipeline,
        gen_engine,
        reward_fns,
        collective,
        device,
        training_context,
        start_iteration,
        common_chkp_kwargs,
    ):
        """Synchronous GRPO loop (Policy and Reference co-located)."""
        logger.info(f"Starting Unified GRPO Loop from iteration {start_iteration}...")
        from optimus_dl.core.log import trange

        pbar = trange(
            start_iteration,
            self.cfg.optimization.iterations,
            initial=start_iteration,
            total=self.cfg.optimization.iterations,
            miniters=self.cfg.common.log_freq,
            disable=not collective.is_local_master,
            smoothing=0,
        )

        for iteration in pbar:
            # 1. Rollout

            try:
                batch = next(data_iter)
            except StopIteration:
                logger.info("Data iterator exhausted.")
                break

            with meters_group("train", log_freq=self.cfg.common.log_freq):
                experience = self._generate_experience(
                    model, ref_model, batch, gen_engine, reward_fns, device
                )

            # 2. Optimize
            self.run_training_iteration(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                train_data_iter=iter(
                    [experience]
                ),  # One-shot iterator for the current batch
                training_context=training_context,
                lr_scheduler=lr_scheduler,
            )

            # 3. Logging
            with meters_group("train", log_freq=self.cfg.common.log_freq) as should_log:
                if should_log:
                    current_metrics = compute_meters(
                        "train",
                        aggregate=True,
                        collective=collective,
                    )

                    if collective.is_local_master:
                        pbar.set_postfix(current_metrics, refresh=False)

                    if collective.is_master:
                        self.log_metrics_to_loggers(current_metrics, iteration, "train")
            step_meters("train")
            reset_meters("train")
            # 4. Evaluation

            eval_needed = self.evaluator.should_run_evaluation(
                iteration, eval_datapipeline
            )
            if eval_needed:
                self.evaluate_and_log(
                    iteration=iteration,
                    model=model,
                    criterion=criterion,
                    eval_datapipeline=eval_datapipeline,
                    collective=collective,
                    device=device,
                )
            # 5. Checkpoint
            if (
                self.cfg.common.save_freq
                and (iteration + 1) % self.cfg.common.save_freq == 0
            ):
                self.save_checkpoint_if_needed(
                    iteration=iteration, **common_chkp_kwargs
                )

            logger.info(f"Iteration {iteration} optimization complete")

    def _generate_experience(
        self, model, ref_model, batch, gen_engine, reward_fns, device
    ) -> dict[str, Any]:
        """Generate rollouts, compute rewards and logprobs."""
        prompt_ids = batch["input_ids"].to(device)
        batch_size = prompt_ids.size(0)
        G = self.cfg.num_generations

        # Expand prompts for group generation
        expanded_prompt_ids = prompt_ids.repeat_interleave(G, dim=0)
        logger.debug(
            f"Expanded prompts for G={G}. Total prompts: {expanded_prompt_ids.shape[0]}"
        )

        with torch.no_grad():
            # Generate
            logger.debug("Rollout: Starting generation...")
            all_ids = gen_engine.generate(
                model, expanded_prompt_ids, self.cfg.generation_config
            )
            logger.debug(
                f"Rollout: Generation complete. all_ids shape: {all_ids.shape}"
            )

            # Masks
            prompt_len = prompt_ids.size(1)
            completion_mask = torch.ones_like(all_ids)
            completion_mask[:, :prompt_len] = 0

            # Ref Logprobs
            logger.debug("Rollout: Computing reference logprobs...")
            ref_outputs = ref_model(all_ids)
            ref_logits = ref_outputs["logits"]

            # Old Policy Logprobs (from the same model that generated)
            logger.debug("Rollout: Computing old policy logprobs...")
            old_outputs = model(all_ids)
            old_logits = old_outputs["logits"]

            # Helper to extract logprobs for selected tokens
            def get_per_token_logprobs(logits, tokens):
                # Logprobs for the tokens that were actually sampled
                # Tokens: [P1, P2, C1, C2, C3]
                # Logits: [L1, L2, L3, L4, L5] -> L1 predicts P2, L2 predicts C1, etc.
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                token_logprobs = torch.gather(
                    log_probs, dim=-1, index=tokens[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                return token_logprobs

            ref_per_token_logprobs = get_per_token_logprobs(ref_logits, all_ids)
            old_per_token_logprobs = get_per_token_logprobs(old_logits, all_ids)

            # Rewards
            prompts = []
            completions = []
            answers = []
            has_answers = "answer" in batch

            if self.tokenizer:
                for i in range(batch_size):
                    p_text = self.tokenizer.decode(prompt_ids[i].tolist())
                    ans_text = batch["answer"][i] if has_answers else ""
                    for g in range(G):
                        prompts.append(p_text)
                        answers.append(ans_text)
                        # Extract only the completion part
                        c_ids = all_ids[i * G + g, prompt_len:].tolist()
                        c_text = self.tokenizer.decode(c_ids)
                        completions.append(c_text)
                        if i == 0 and g == 0:
                            logger.debug(
                                f"Rollout Sample 0:\nPrompt: {p_text[:100]}...\nCompletion: {c_text[:100]}..."
                            )
            else:
                logger.warning(
                    "No tokenizer found in GRPORecipe, using placeholders for rewards."
                )
                prompts = ["prompt placeholder"] * (batch_size * G)
                completions = ["completion placeholder"] * (batch_size * G)
                answers = ["answer placeholder"] * (batch_size * G)

            rewards = []
            for r_fn in reward_fns:
                rewards.append(
                    r_fn(prompts=prompts, completions=completions, answers=answers).to(
                        device
                    )
                )

            # Sum rewards across functions and compute advantages
            total_rewards = torch.stack(rewards).sum(dim=0)  # (batch_size * G)
            mean_reward = total_rewards.mean()
            log_averaged(
                "reward", lambda: mean_reward.item(), weight=batch_size, round=4
            )

            # Group-relative advantages
            # Reshape to (batch_size, G) to compute mean/std within groups
            grouped_rewards = total_rewards.view(batch_size, G)
            mean = grouped_rewards.mean(dim=1, keepdim=True)
            std = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (grouped_rewards - mean) / std
            advantages = advantages.view(-1)  # Flatten back to (batch_size * G)
            max_advantage = advantages.max()
            log_averaged(
                "advantage_max",
                lambda: max_advantage.item(),
                weight=batch_size,
                round=4,
            )

        return {
            "input_ids": all_ids,
            "completion_mask": completion_mask,
            "old_logprobs": old_per_token_logprobs,
            "ref_logprobs": ref_per_token_logprobs,
            "advantages": advantages,
        }
