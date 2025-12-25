import logging
import os

import torch

from optimus_dl.core.bootstrap import bootstrap_module
from optimus_dl.modules.distributed.base import Collective
from optimus_dl.modules.distributed.config import DistributedConfig
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.distributed.mesh import MeshCollective

logger = logging.getLogger(__name__)

_collective: Collective | None = None


def build_best_collective(
    config: DistributedConfig, device: torch.device | None = None
) -> Collective:
    global _collective
    if _collective is not None:
        return _collective

    logger.info(f"Initializing collective {config}")
    needed_envs = ("LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK", "WORLD_SIZE")

    distr = True
    for env in needed_envs:
        if env not in os.environ:
            distr = False
            logger.info(f"{env} is not defined, building not distributed collective")
            break

    if distr:
        single_rank = int(os.environ["WORLD_SIZE"]) <= 1
        if single_rank:
            distr = False
            logger.info("Single rank detected, building not distributed collective")

    if distr:
        device_type = "cpu"
        if device is not None:
            device_type = device.type
        elif torch.cuda.is_available():
            device_type = "cuda"

        collective = MeshCollective(
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            device_type=device_type,
            tp_size=config.tp_size,
            sharding_world_size=config.sharding_world_size,
        )
    else:
        # Pass device_type to FakeCollective as well
        if device is not None:
            device_type = device.type
        elif torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

        collective = FakeCollective(rank=0, world_size=1, device_type=device_type)
    logger.info(f"Built collective: {collective}")
    return collective


bootstrap_module(__name__)
