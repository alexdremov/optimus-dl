import gc
import logging
import os

import psutil

logger = logging.getLogger(__name__)


def force_kill_children(timeout_seconds: int = 15, max_retries=10) -> None:
    """
    Recursively terminates all child processes of the current process.
    """
    # try to gracefully terminate joblib
    force_terminate_joblib()
    finish_wandb()

    children = ["dummy"]
    retries = 0

    while len(children) > 0 and retries < max_retries:
        retries += 1
        gc.collect()
        try:
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
        except Exception:
            logger.warning(
                f"Failed to get child processes for PID {os.getpid()}", exc_info=True
            )
            return

        # Phase 1: Polite termination
        for child in children:
            try:
                child.terminate()
            except Exception:
                logger.warning(
                    f"Failed to terminate child process: {child.pid}", exc_info=True
                )
                continue

        # Phase 2: Force kill if they ignore SIGTERM
        _, alive = psutil.wait_procs(children, timeout=timeout_seconds)
        for child in alive:
            try:
                logger.warning(f"Force killing unresponsive child process: {child.pid}")
                child.kill()
            except Exception:
                logger.warning(
                    f"Failed to force kill child process: {child.pid}", exc_info=True
                )
                continue


def force_terminate_joblib():
    """
    Forcefully terminates Joblib workers without allowing them to respawn.
    """
    try:
        import joblib.externals.loky.reusable_executor as loky_reusable_executor
    except ImportError:
        logger.warning(
            "Joblib is not installed, cannot force terminate Joblib workers."
        )
        return

    executor = loky_reusable_executor._executor
    if executor is not None:
        logger.info("Shutting down Joblib reusable executor to prevent worker respawn.")
        logger.warning(
            "Lokky workers may not terminated completely. This may lead to the process hanging if they do not exit on their own. It is recomended to not use joblib."
        )
        executor.shutdown(wait=False, kill_workers=True)


def finish_wandb():
    """
    Ensures that all WandB processes are terminated to prevent hanging.
    """
    try:
        import wandb
    except ImportError:
        logger.warning("WandB is not installed, cannot finish WandB run.")
        return

    if wandb.run is not None:
        logger.info("Finishing WandB run to ensure all processes are terminated.")
        wandb.finish()
