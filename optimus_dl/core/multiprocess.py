import gc
import logging
import os

import psutil


def force_kill_children(timeout_seconds: int = 15) -> None:
    """
    Recursively terminates all child processes of the current process.
    """
    # try to gracefully terminate joblib
    force_terminate_joblib()
    finish_wandb()

    gc.collect()

    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
    except psutil.NoSuchProcess:
        return

    # Phase 1: Polite termination
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue

    # Phase 2: Force kill if they ignore SIGTERM
    gone, alive = psutil.wait_procs(children, timeout=timeout_seconds)
    for child in alive:
        try:
            logging.warning(f"Force killing unresponsive child process: {child.pid}")
            child.kill()
        except psutil.NoSuchProcess:
            continue


def force_terminate_joblib():
    """
    Forcefully terminates Joblib workers without allowing them to respawn.
    """
    try:
        import joblib.externals.loky.reusable_executor as loky_reusable_executor
    except ImportError:
        logging.warning(
            "Joblib is not installed, cannot force terminate Joblib workers."
        )
        return

    executor = loky_reusable_executor._executor
    if executor is not None:
        logging.info(
            "Shutting down Joblib reusable executor to prevent worker respawn."
        )
        executor.shutdown(wait=False, kill_workers=True)


def finish_wandb():
    """
    Ensures that all WandB processes are terminated to prevent hanging.
    """
    try:
        import wandb
    except ImportError:
        logging.warning("WandB is not installed, cannot finish WandB run.")
        return

    if wandb.run is not None:
        logging.info("Finishing WandB run to ensure all processes are terminated.")
        wandb.finish()
